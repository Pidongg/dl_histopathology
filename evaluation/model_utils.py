import time
import numpy as np
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

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=False, max_width=2000, max_height=2000, get_unknowns=False,
                        classes=None, agnostic=False):
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
        box = xywh2xyxy(x_all[:, 0, :4])

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
            # Getting the indices where classes are above certain conf threshold
            i, j = (x_all[:, 0, 4:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x_all[i, 0, j + 4].unsqueeze(1), j.float().unsqueeze(1)), 1)
            x_all = x_all[i, ...]
        else:  # best class only
            conf, j = x_all[:, 0, 4:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]
            x_all = x_all[conf > conf_thres, ...]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        # c will contain the class for each detection i
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        # boxes will contain all the detections bboxes (shape i X 4), and those will be passed to torchvision NMS operator
        # In order to avoid removing bboxes with big iou but classifying different classes, the bounding boxes are shifted/offseted
        # This offset is done by "c.view(-1, 1) * max_wh", which multiplies each discrete class by the mximum width/height allowed,
        #    (notice that detections with coordinates above this maximum value were dropped out in the beginning of this function)
        #    this way,we are able to clearly separate all the bounding boxes in a way that bboxes with similar coordinates but
        #    with different classes are never clustered together for the nms() function
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        
        # This "merge" won't happen because of beginning of this function and later changes
        #if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #    try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #        iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #        weights = iou * scores[None]  # box weights, None to index a tensor basically adds a new dimension
        #        x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #        # i = i[iou.sum(1) > 1]  # require redundancy
        #    except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
        #        print('ERROR at non_max_suppression merge:', x, i, x.shape, i.shape)
        #        pass

        output[xi] = x[i]
        
        
        x_all = x_all[i, :, :]
        all_scores[xi] = x_all[:, 0, 4:]

        if prediction.shape[2] > 1:
            sampled_coords[xi] = x_all[:, 1:, :4]
        
        # I've removed time_limit to guarantee that everything is processed
        #if (time.time() - t) > time_limit:
        #    break  # time limit exceeded

    return output, all_scores, sampled_coords