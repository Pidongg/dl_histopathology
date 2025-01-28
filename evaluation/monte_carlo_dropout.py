import os
import torch
import json
from ultralytics.utils import LOGGER
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preparation')))
import data_utils
from model_utils import non_max_suppression

def enable_dropout(model):
    """Enable dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def monte_carlo_predictions(model, img, conf_thresh, iou_thresh, num_samples=30):
    """
    Perform multiple forward passes with dropout enabled.
    Returns raw model outputs before NMS.
    """
    infs_all = []
    num_classes = 4 
    
    # Perform multiple forward passes
    with torch.no_grad():
        for _ in range(num_samples):
            # Enable dropout and set to eval mode
            model.eval()
            enable_dropout(model)

            # Get raw model output (before NMS)
            preds = model.predict(str(img), conf=conf_thresh, skip_nms=True)[0]
            # nms requires preds to change from (batch_size, num_classes + 4 + num_masks, num_boxes) to (batch_size, num_boxes, num_classes + 4)
            preds = preds[:, :num_classes+4, :]
            preds = preds.transpose(-2, -1)
            infs_all.append(preds.unsqueeze(2))
            
        if num_samples > 1:
            inf_mean = torch.mean(torch.stack(infs_all), dim=0)
            infs_all.insert(0, inf_mean)
            inf_out = torch.cat(infs_all, dim=2)
        else:
            inf_out = infs_all[0]
        
        # Apply NMS and get sampled coordinates
        output, all_scores, sampled_coords = non_max_suppression(
            inf_out,
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            multi_label=False,
            max_width=512,
            max_height=512
        )
        # Process sampled coordinates if we have multiple samples
        if num_samples > 1 and output[0] is not None and len(output[0]) > 0:
            # Use fixed dimensions
            height, width = 512, 512
            
            # Process sampled coordinates for the first (and only) image in batch
            sampled_boxes = data_utils.xywh2xyxy(sampled_coords[0].reshape(-1, 4)).reshape(sampled_coords[0].shape)
            data_utils.clip_coords(sampled_boxes.reshape(-1, 4), (height, width))
            
            # Calculate covariances for each detection
            covariances = torch.zeros(sampled_boxes.shape[0], 2, 2, 2, device=output[0].device)
            print(sampled_boxes)
            for det_idx in range(sampled_boxes.shape[0]):
                covariances[det_idx, 0, ...] = data_utils.cov(sampled_boxes[det_idx, :, :2])
                covariances[det_idx, 1, ...] = data_utils.cov(sampled_boxes[det_idx, :, 2:])
            
            # Ensure covariances are positive semi-definite
            for det_idx in range(len(covariances)):
                for i in range(2):
                    cov_matrix = covariances[det_idx, i].cpu().numpy()
                    if not data_utils.is_pos_semidef(cov_matrix):
                        LOGGER.warning('Converting covariance matrix to near PSD')
                        cov_matrix = data_utils.get_near_psd(cov_matrix)
                        covariances[det_idx, i] = torch.tensor(cov_matrix, device=output[0].device)
            
            # Round values for smaller size
            covariances = torch.round(covariances * 1e6) / 1e6
        else:
            covariances = None
        # # Take first batch element since we're processing one image at a time
        # output = output[0] if output else None
        # all_scores = all_scores[0] if all_scores else None
    return output, all_scores, covariances

def save_mc_predictions_to_json(model, data_yaml, conf_thresh, save_path, num_samples=30, iou_thresh=0.6):
    """Save Monte Carlo predictions to JSON in same format as YOLO predictions."""
    results_dict = {}
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    val_path = os.path.join(config.get('path', ''), config.get('val', ''))
    if not os.path.exists(val_path):
        LOGGER.error(f"Invalid validation path: {val_path}")
        return
    
    # Process all images in all subdirectories
    image_files = []
    for root, _, _ in os.walk(val_path):
        image_files.extend(data_utils.list_files_of_a_type(root, '.png'))
    
    LOGGER.info(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        try:
            # Get processed predictions
            output, all_scores, covariances = monte_carlo_predictions(
                model,
                img_file,
                conf_thresh,
                iou_thresh,
                num_samples
            )
            print(output, all_scores, covariances)
            # Get relative path for image key
            rel_path = os.path.relpath(img_file, val_path)
            image_dets = []
            # Format predictions for current image
            if output is not None and len(output) > 0:
                output = output[0]
                all_scores = all_scores[0]
                for idx, pred in enumerate(output):
                    print(pred)
                    box = pred[:4].cpu().numpy()
                    conf = float(pred[4].cpu().numpy())
                    cls_id = int(pred[5].cpu().numpy())
                    class_confs = all_scores[idx].cpu().numpy()
                    
                    det_dict = {
                        "boxes": box.tolist(),
                        "conf": conf,
                        "cls_id": cls_id,
                        "class_confs": class_confs.tolist()
                    }
                    if covariances is not None:
                        det_dict["covars"] = covariances[idx].cpu().numpy().tolist()
                    print(det_dict)
                    image_dets.append(det_dict)
            
            # Add predictions for this image to results dict
            results_dict[rel_path] = image_dets
            
        except Exception as e:
            LOGGER.warning(f"Error processing image {img_file}: {e}")
            results_dict[os.path.relpath(img_file, val_path)] = []
            continue
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return results_dict