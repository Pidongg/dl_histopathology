import time
from data_preparation import data_utils
import numpy as np
from pdq_evaluation.read_files import LOGGER
import torch    
import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def enable_dropout(model):
    """Enable dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.6, multi_label=False, max_width=2000, max_height=2000, get_unknowns=False,
                        classes=None, agnostic=False, use_xyxy_format=False, class_conf_thresholds=None):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        n X 6: n X (x1, y1, x2, y2, conf, cls)
    """
    
    # prediction has shape (batch_size X detections X 8) for without mc-dropout or (batch_size X detections X (1 + sampled_tensors) X 8) otherwise
    # The first 4 values are bbox coordinates, the remaining 4 are class confidences
    
    # If dimension is 3, adding dummy dimension to comply with rest of code
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(2)
    
    # Settings
    # In ultralytics' repository "merge" is True, but I've changes this to False, some discussion here: https://github.com/ultralytics/yolov3/issues/679
    merge = False  # merge for best mAP
    min_wh, max_wh = 0, 4096  # (pixels) minimum and maximum box width and height
    #time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[-1] - 4  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    all_scores = [None] * prediction.shape[0]
    if prediction.shape[2] > 1:
        sampled_coords = [None] * prediction.shape[0]
    else:
        sampled_coords = None
    for xi, x_all in enumerate(prediction):  # image index, image inference
        
        # Get max confidence across all classes as the main confidence score
        conf_scores = x_all[:, 0, 4:].max(dim=1)[0]
        # Apply constraints
        x_all = x_all[conf_scores > conf_thres]  # confidence
        x_all = x_all[((x_all[:, 0, 2:4] > min_wh) & (x_all[:, 0, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        if not x_all.shape[0]:
            continue

        # No need to compute conf since we don't have obj_conf anymore
        x_all_orig_shape = x_all.shape
        x_all = x_all.reshape(-1, x_all.shape[2])
        x_all = x_all.reshape(x_all_orig_shape)

        if get_unknowns:
            # Getting bboxes only when all the labels are predicted with prob below 0.5
            x_all = x_all[(x_all[:, 0, 4:] < 0.5).all(1) & (x_all[:, 0, 4:] > 0.1).any(1)]
            #x_all = x_all[(x_all[:, 0, 5:] < 0.5).all(1)]
            #x_all = x_all[(x_all[:, 0, 5:] < 0.2).all(1) & (x_all[:, 0, 5:] > 0.1).any(1)]
            if x_all.shape[0] == 0:
                continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        if not use_xyxy_format:
            box = xywh2xyxy(x_all[:, 0, :4])
        else:
            # If already in xyxy format, just use as is
            box = x_all[:, 0, :4]

        # Removing bboxes out of image limits
        wrong_bboxes = (box < 0).any(1) | (box[:, [0, 2]] >= max_width).any(1) | (box[:, [1, 3]] >= max_height).any(1)
        box = box[~wrong_bboxes]
        x_all = x_all[~wrong_bboxes]

        # If none remain process next image
        n = x_all.shape[0]  # number of boxes
        if not n:
            continue

        # The next parts of code will filter labels according to confidence threshold,
        #   so, if get_unknowns is True, just get everything
        if get_unknowns:
            conf_thres = 0

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            if class_conf_thresholds is not None:
                # Apply class-specific thresholds
                mask = torch.zeros_like(x_all[:, 0, 4:], dtype=torch.bool)
                for c, thresh in enumerate(class_conf_thresholds):
                    if c < mask.shape[1]:  # Ensure we don't exceed number of classes
                        mask[:, c] = x_all[:, 0, 4 + c] > thresh
                i, j = mask.nonzero().t()
            else:
                # Use global threshold
                i, j = (x_all[:, 0, 4:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x_all[i, 0, j + 4].unsqueeze(1), j.float().unsqueeze(1)), 1)
            x_all = x_all[i, ...]
        else:  # best class only
            conf, j = x_all[:, 0, 4:].max(1)
            
            if class_conf_thresholds is not None:
                # Check if each detection's best class passes its specific threshold
                class_thresholds = torch.tensor(class_conf_thresholds, device=conf.device)
                # Get threshold for each detection based on its class
                thresholds = class_thresholds[j.long()]
                # Filter based on class-specific thresholds
                mask = conf > thresholds
            else:
                # Use global threshold
                mask = conf > conf_thres
                
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[mask]
            x_all = x_all[mask, ...]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        
        output[xi] = x[i]
        
        
        x_all = x_all[i, :, :]
        all_scores[xi] = x_all[:, 0, 4:]

        if prediction.shape[2] > 1:
            sampled_coords[xi] = x_all[:, 1:, :4]

    return output, all_scores, sampled_coords

def monte_carlo_predictions(self, img, conf_thresh, iou_thresh, num_samples=30):
    """
    Perform multiple forward passes with dropout enabled.
        Returns raw model outputs before NMS.
        """
    infs_all = []

    # Perform multiple forward passes
    with torch.no_grad():
        for _ in range(num_samples):
            # Get raw model output
            _, predictions = self.infer_for_one_img(img)
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            infs_all.append(predictions.unsqueeze(2))
        
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
            max_width=800,
            max_height=800
        )
        image_width = 512  # Or get actual image dimensions
        scale_factor = image_width / 800

        # Process sampled coordinates if we have multiple samples
        if num_samples > 1 and output[0] is not None and len(output[0]) > 0:
            # Use fixed dimensions
            height, width = 512, 512
            output[0][:, :4] *= scale_factor
            
            # Process sampled coordinates for the first (and only) image in batch
            sampled_boxes = data_utils.xywh2xyxy(sampled_coords[0].reshape(-1, 4)).reshape(sampled_coords[0].shape)
            sampled_boxes *= scale_factor
            data_utils.clip_coords(sampled_boxes.reshape(-1, 4), (height, width))

            # Calculate covariances for each detection
            covariances = torch.zeros(sampled_boxes.shape[0], 2, 2, 2, device=output[0].device)
            for det_idx in range(sampled_boxes.shape[0]):
                covariances[det_idx, 0, ...] = data_utils.cov(sampled_boxes[det_idx, :, :2]) # covariance of top left corner
                covariances[det_idx, 1, ...] = data_utils.cov(sampled_boxes[det_idx, :, 2:]) # covariance of bottom right corner

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