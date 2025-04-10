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

def check_dropout_state(model, msg=""):
    """Check if dropout layers are in training mode."""
    dropout_states = []
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            dropout_states.append(m.training)
    if dropout_states:
        LOGGER.info(f"{msg} Dropout layers training state: {dropout_states}")
    else:
        LOGGER.warning("No dropout layers found in model!")
    return dropout_states


def monte_carlo_predictions(model, img, conf_thresh, iou_thresh, num_samples=30, is_yolo=False, input_size=640, class_conf_thresholds=None):
    """
    Perform multiple forward passes with dropout enabled.
    Returns raw model outputs before NMS.
    
    Args:
        model: YOLO model
        img: Input image path
        conf_thresh: Confidence threshold (float or list of floats for class-specific thresholds)
        iou_thresh: IoU threshold for NMS
        num_samples: Number of Monte Carlo samples
        is_yolo: Whether using YOLO model (to enable dropout)
        input_size: Input size for the model
    """
    infs_all = []
    num_classes = 4 
    
    # Perform multiple forward passes
    with torch.no_grad():
        for i in range(num_samples):
            if is_yolo:
                enable_dropout(model)
            
            # Get raw model output (before NMS)
            preds = model.predict(str(img), conf=conf_thresh, skip_nms=True, iou=iou_thresh)[0]
            
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
        # Use model's native dimensions for NMS
        model_height, model_width = input_size, input_size

        # Clip coordinates before NMS
        for batch_idx in range(inf_out.shape[0]):
            for sample_idx in range(inf_out.shape[2]):
                # Convert from xywh to xyxy for clipping
                boxes_xyxy = data_utils.xywh2xyxy(inf_out[batch_idx, :, sample_idx, :4].clone())
                # Clip to image boundaries
                data_utils.clip_coords(boxes_xyxy, (model_height, model_width))
                # For now, we'll directly use the clipped xyxy in non_max_suppression
                inf_out[batch_idx, :, sample_idx, :4] = boxes_xyxy
                # Set a flag to indicate we're now using xyxy format
                use_xyxy_format = True
        
        output, all_scores, sampled_coords = non_max_suppression(
            inf_out,
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            multi_label=False,
            max_width=model_width,
            max_height=model_height,
            use_xyxy_format=use_xyxy_format,
            class_conf_thresholds=class_conf_thresholds
        )
        
        # Calculate scaling factor dynamically
        image_width = 512  # Or get actual image dimensions
        scale_factor = image_width / model_width
        
        # Process sampled coordinates if we have multiple samples
        if num_samples > 1 and output[0] is not None and len(output[0]) > 0:
            # Scale output boxes from model size (640x640) to image size (512x512)
            output[0][:, :4] *= scale_factor
            
            # Use fixed dimensions for final image
            height, width = 512, 512
            
            # Process sampled coordinates for the first (and only) image in batch
            sampled_boxes = data_utils.xywh2xyxy(sampled_coords[0].reshape(-1, 4)).reshape(sampled_coords[0].shape)
            
            # Scale sampled boxes from model size to image size
            sampled_boxes *= scale_factor
            
            # Clip coordinates to image boundaries
            data_utils.clip_coords(sampled_boxes.reshape(-1, 4), (height, width))
            
            # Calculate covariances for each detection (using scaled coordinates)
            covariances = torch.zeros(sampled_boxes.shape[0], 2, 2, 2, device=output[0].device)
            for det_idx in range(sampled_boxes.shape[0]):
                top_left_cov = data_utils.cov(sampled_boxes[det_idx, :, :2])
                bottom_right_cov = data_utils.cov(sampled_boxes[det_idx, :, 2:])
                
                covariances[det_idx, 0, ...] = top_left_cov
                covariances[det_idx, 1, ...] = bottom_right_cov
            
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
        elif output[0] is not None and len(output[0]) > 0:
            # If we only have one sample but still have valid output, apply scaling
            output[0][:, :4] *= scale_factor
            covariances = None
        else:
            covariances = None
            
    return output, all_scores, covariances

def save_mc_predictions_to_json(model, data_yaml, conf_thresh, save_path, num_samples=30, iou_thresh=0.6, is_yolo=False, input_size=640, class_conf_thresholds=None):
    """
    Save Monte Carlo predictions to JSON in same format as YOLO predictions.
    
    Args:
        model: YOLO model
        data_yaml: Path to data configuration YAML file
        conf_thresh: Confidence threshold (float or list of floats for class-specific thresholds)
        save_path: Path to save predictions JSON
        num_samples: Number of Monte Carlo samples
        iou_thresh: IoU threshold for NMS
        is_yolo: Whether the model is YOLO (for enabling dropout)
        input_size: Input size for the model
    """
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
    image_files.sort()
    LOGGER.info(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        try:
            # Get processed predictions
            output, all_scores, covariances = monte_carlo_predictions(
                model,
                img_file,
                conf_thresh,
                iou_thresh,
                num_samples,
                is_yolo=is_yolo,
                input_size=input_size,
                class_conf_thresholds=class_conf_thresholds
            )
            
            # Get the full filename without any parent directories
            rel_path = os.path.basename(img_file)
            
            # If the filename starts with a directory that matches its name, remove it
            # e.g., "747331/747331 [...]" becomes "747331 [...]"
            if '/' in rel_path:
                parts = rel_path.split('/')
                if parts[-2] in parts[-1]:
                    rel_path = parts[-1]
            
            image_dets = []
            # Format predictions for current image
            if output is not None and len(output) > 0:
                output = output[0]
                all_scores = all_scores[0]
                for idx, pred in enumerate(output):
                    box = pred[:4].cpu().numpy()
                    conf = float(pred[4].cpu().numpy())
                    cls_id = int(pred[5].cpu().numpy())
                    class_confs = all_scores[idx].cpu().numpy()
                    
                    # Apply class-specific threshold if provided
                    if isinstance(conf_thresh, list) and cls_id < len(conf_thresh):
                        class_threshold = conf_thresh[cls_id]
                        if conf < class_threshold:
                            continue
                    
                    det_dict = {
                        "boxes": box.tolist(),
                        "conf": conf,
                        "cls_id": cls_id,
                        "class_confs": class_confs.tolist()
                    }
                    if covariances is not None:
                        det_dict["covars"] = covariances[idx].cpu().numpy().tolist()
                    image_dets.append(det_dict)
            
            # Add predictions for this image to results dict
            results_dict[rel_path] = image_dets
            
        except Exception as e:
            LOGGER.warning(f"Error processing image {img_file}: {e}")
            results_dict[rel_path] = []
            continue
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return results_dict