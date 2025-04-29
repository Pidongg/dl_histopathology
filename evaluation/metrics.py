# Computing validation metrics for models.
# Where explicitly noted, this file contains code based on Ultralytics' YOLOv8 source code, available at
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py
# under the AGPL-3.0 license, which therefore also covers this code.
# The full text of the license may be found at https://www.gnu.org/licenses/agpl-3.0.txt.
# (last updated 07/05/2024)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import os

def upcast(t: Tensor) -> Tensor:
    """
    Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    From https://github.com/pytorch/vision/blob/main/torchvision/ops/_utils.py
    """
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of N bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. (x1, y1, x2, y2) format expected.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes in (x1, y1, x2, y2) format
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # get intersecting box
    x1y1 = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # coords of top left corner: shape [N,M,2]
    x2y2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # coords of bottom right corner: shape [N,M,2]

    wh = upcast(x2y2 - x1y1).clamp(min=0)  # prevent lengths < 0. shape [N,M,2]: width, height per box pair
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]: area per box pair

    union = area1[:, None] + area2 - inter

    eps = 1e-7  # prevent division by 0

    iou = inter / (union + eps)
    return iou


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.
    Based on https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py

    Args:
        recall (list): The recall curve as a list of recall values for confidence.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))  # remember that recall is stored in order of desc confidence
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute Pr_interp, i.e. precision at each conf level that gives a monotonic curve.
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def smooth(y, f=0.05):
    """
    Box filter of fraction f.

    Implementation from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py
    """
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def plot_curve(px, py, save_dir, names: dict[int, str], xlabel, ylabel, show_max=False):
    """
    Plots a curve.
    Adapted from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py

    Takes:
        show_max (bool): If true, show the x-value that maximises y.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    
    # Handle empty py
    if len(py) == 0:
        print(f"Warning: No data found to plot {ylabel}-{xlabel} curve. Skipping.")
        plt.close(fig)
        return
    
    try:
        avg_y = smooth(py.mean(0), 0.05)  # smoothed curve averaging each class' curve
        max_i = avg_y.argmax()  # index that maximises avg_y
    except Exception as e:
        print(f"Warning: Error calculating average curve: {e}. Skipping.")
        plt.close(fig)
        return

    class_idxs = sorted(list(names.keys()))

    for i, y in enumerate(py):
        if i in class_idxs:
            curve_label = f"{names[i]}" + (f"({y[max_i]:.2f} at {px[max_i]:.3f})" if show_max else "")
            ax.plot(px, y, linewidth=1, label=curve_label)
            # ax.plot(px, y, linewidth=1, label=f"{names[i]}")

    avg_curve_label = f"all classes" + (f"(max {avg_y.max():.2f} at {px[max_i]:.3f})" if show_max else "")
    ax.plot(px, avg_y, linewidth=3, label=avg_curve_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(f"{ylabel}-{xlabel} Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_pr_curve(px, py, ap, iou, save_dir="pr_curve.png", names=()):
    """
    Plots a precision-recall curve.
    Adapted from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    
    # Handle empty py list
    if len(py) == 0:
        print("Warning: No precision values found to plot PR curve. Skipping PR curve plot.")
        plt.close(fig)
        return
    
    try:
        py = np.stack(py, axis=1)
    except ValueError as e:
        print(f"Warning: Could not stack precision values: {e}. Skipping PR curve plot.")
        plt.close(fig)
        return

    class_idxs = sorted(list(names.keys()))

    for i, y in enumerate(py.T):
        if i in class_idxs:
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i]:.3f}")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@%.3f" % (ap.mean(), iou))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


class ObjectDetectionMetrics:
    """
    This class is for computing detection metrics such as mean average precision (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        idx_to_name (Dict[int, str]): A dictionary that maps class indexes to names.
        detections (List[Tensor]): A list of detections per image, each in (x1, y1, x2, y2, conf, class) format.
        ground_truths (List[Tensor]): A list of ground truth labels per image, each in (x1, y1, x2, y2, class) format.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        idx_to_name (Dict[int, str]): A dictionary that maps class indexes to names.
        num_classes (int): Number of classes

    Methods:
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
    """
    def __init__(self, save_dir, idx_to_name, detections: list[Tensor], ground_truths: list[Tensor],
                 device):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.idx_to_name = idx_to_name
        self.num_classes = len(idx_to_name.keys())
        self.device = device
        self.iou_threshold = torch.linspace(0.5, 0.95, 10)
        self.detections = detections
        self.ground_truths = ground_truths

        self.pred_true = None
        self.conf = None
        self.pred_cls = None
        self.gt_cls = None
        self.gt_per_pred_all = None
        self.pred_per_gt_all = None

        self.match_predictions()

        # shape: (num_iou_thresholds, num_classes)
        self.ap = None

    def match_predictions(self):
        """
        Matches `self.detections` to `self.ground_truths` for each image and for each iou_threshold specified.

        Returns:
            detection_true (Tensor[t, n]): Binary values indicating whether each detection is correct (1) or false (0),
                for each value in iou_threshold.
            conf (Tensor[n]): Confidence for each detection.
            detection_cls (Tensor[n]): Class of each detection.
            gt_cls (Tensor[m]): Class of each ground truth label.
            gt_per_pred (Tensor[n]): Holds the value of the gt matching each pred, and -1 for no matches.
            pred_per_gt (Tensor[n]): Holds the value of the pred matching each gt, and -1 for no matches.
        """
        t = self.iou_threshold.shape[0]
        pred_true_all = []
        conf_all = []
        pred_cls_all = []
        gt_cls_all = []
        gt_per_pred_all = []
        pred_per_gt_all = []

        # counters to hold the number of preds and ground truths seen so far
        #   to store gt_per_pred_all and pred_per_gt_all correctly
        num_preds_curr = 0
        num_gt_curr = 0

        # Iterate over each detections-labels pair (each corresponding to an image)
        for i in range(len(self.detections)):
            pred = self.detections[i]
            gt = self.ground_truths[i]
            
            # Skip if either pred or gt is None or empty
            if pred is None or gt is None or len(pred) == 0 or len(gt) == 0:
                continue
            
            # Ensure pred and gt are tensors
            if not isinstance(pred, torch.Tensor):
                try:
                    pred = torch.tensor(pred, device=self.device)
                except:
                    print(f"Warning: Could not convert prediction to tensor. Skipping image {i}.")
                    continue
                
            if not isinstance(gt, torch.Tensor):
                try:
                    gt = torch.tensor(gt, device=self.device)
                except:
                    print(f"Warning: Could not convert ground truth to tensor. Skipping image {i}.")
                    continue
            
            # Ensure pred and gt have the expected dimensions
            if len(pred.shape) != 2 or len(gt.shape) != 2:
                print(f"Warning: Unexpected dimensions for pred or gt at index {i}. Expected 2D tensors.")
                continue
                
            # Make sure we have enough dimensions for indexing
            if pred.shape[1] <= 5 or gt.shape[1] <= 4:
                print(f"Warning: Not enough columns in pred or gt at index {i}. Skipping.")
                continue
                
            pred_cls = pred[:, -1]
            gt_cls = gt[:, -1]
            conf = pred[:, 4]

            n = pred.shape[0]
            m = gt.shape[0]

            # append conf and class information to the full lists
            conf_all.append(conf)
            pred_cls_all.append(pred_cls)
            gt_cls_all.append(gt_cls)

            # calculate pairwise iou between pred and gt boxes
            try:
                iou = box_iou(pred[:, :4], gt[:, :4])
            except Exception as e:
                print(f"Error calculating IoU for image {i}: {str(e)}")
                print(f"pred shape: {pred.shape}, gt shape: {gt.shape}")
                print(f"pred[:, :4] shape: {pred[:, :4].shape}, gt[:, :4] shape: {gt[:, :4].shape}")
                continue

            # calculate pred_true for each prediction.
            # tensor[t, num_preds]
            correct = np.zeros((t, pred_cls.shape[0]))

            # LxD matrix where L - labels, D - detections
            try:
                correct_class = gt_cls[:, None] == pred_cls
                iou = iou.T
                iou = iou * correct_class  # zero out the wrong classes
                iou = iou.cpu().numpy()
            except Exception as e:
                print(f"Error processing classes for image {i}: {str(e)}")
                continue
                
            # pred_per_gt: tensor[t, num_gt]
            pred_per_gt = np.zeros((t, m)) - 1

            # gt_per_pred: tensor[t, num_pred]
            gt_per_pred = np.zeros((t, n)) - 1

            # iterate over each iou threshold
            for i, threshold in enumerate(self.iou_threshold.cpu().tolist()):
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T

                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        # store the (gt, pred) pairs in descending order of iou
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        
                        # which prediction is matched to which gt
                        pred_per_gt[i, matches[:, 0]] = matches[:, 1] + num_preds_curr
                        
                        # which gt is matched to which prediction
                        gt_per_pred[i, matches[:, 1]] = matches[:, 0] + num_gt_curr

                    correct[i, matches[:, 1].astype(int)] = 1

            pred_true_all.append(torch.tensor(correct, dtype=torch.int, device=self.device))

            pred_per_gt_all.append(torch.tensor(pred_per_gt, dtype=torch.int, device=self.device))
            num_preds_curr += n
            gt_per_pred_all.append(torch.tensor(gt_per_pred, dtype=torch.int, device=self.device))
            num_gt_curr += m

        # Handle case where we have no valid predictions
        if not pred_true_all:
            print("Warning: No valid predictions found. Returning empty results.")
            self.pred_true = torch.zeros((t, 0), device=self.device, dtype=torch.int)
            self.gt_per_pred_all = torch.zeros((t, 0), device=self.device, dtype=torch.int)
            self.pred_per_gt_all = torch.zeros((t, 0), device=self.device, dtype=torch.int)
            self.conf = torch.zeros(0, device=self.device)
            self.pred_cls = torch.zeros(0, device=self.device, dtype=torch.int)
            self.gt_cls = torch.zeros(0, device=self.device, dtype=torch.int)
            return

        # Concatenate the return lists into tensors
        self.pred_true = torch.cat(pred_true_all, axis=-1)
        self.gt_per_pred_all = torch.cat(gt_per_pred_all, axis=-1)
        self.pred_per_gt_all = torch.cat(pred_per_gt_all, axis=-1)

        # Concatenate conf and class info
        self.conf = torch.cat(conf_all, axis=-1)
        self.pred_cls = torch.cat(pred_cls_all, axis=-1)
        self.gt_cls = torch.cat(gt_cls_all, axis=-1)

    def get_confusion_matrix(self, conf_threshold, all_iou=False, prefix="", plot=False, class_conf_thresholds=None):
        """
        Returns a confusion matrix with predicted class as row index, and gt class as col index.
        Matches Ultralytics' implementation.

        Args:
            conf_threshold: Confidence threshold for predictions (used as default)
            all_iou (bool): Indicates whether to compute confusion matrices for all iou thresholds. If false,
                just compute for IOU=0.5.
            prefix (str): Prefix for saved plot filename
            plot (bool): Whether to save confusion matrix plot
            class_conf_thresholds (dict): Dictionary mapping class indices to specific confidence thresholds.
                                         If provided, overrides conf_threshold for specific classes.
        """
        t = self.iou_threshold.shape[0]

        if all_iou:
            iou_thresholds = range(t)
        else:
            iou_thresholds = [0]

        matrix = torch.zeros((len(iou_thresholds), self.num_classes + 1, self.num_classes + 1), dtype=torch.int32)

        for ti in iou_thresholds:
            iou_threshold = self.iou_threshold[ti]
            
            # Process each image's detections and ground truths
            for i in range(len(self.detections)):
                pred = self.detections[i]
                gt = self.ground_truths[i]
                
                # Filter by confidence - handle class-specific thresholds if provided
                if class_conf_thresholds is not None:
                    # Create a mask for predictions to keep
                    mask = torch.zeros(pred.shape[0], dtype=torch.bool, device=pred.device)
                    
                    # Process each class separately with its specific threshold
                    for cls_idx, cls_thresh in class_conf_thresholds.items():
                        cls_mask = (pred[:, 5] == cls_idx) & (pred[:, 4] >= cls_thresh)
                        mask = mask | cls_mask
                    
                    # For classes not in class_conf_thresholds, use the default threshold
                    default_mask = ~torch.tensor([pred[i, 5].item() in class_conf_thresholds for i in range(pred.shape[0])], 
                                               device=pred.device, dtype=torch.bool)
                    default_mask = default_mask & (pred[:, 4] >= conf_threshold)
                    mask = mask | default_mask
                    
                    pred = pred[mask]
                else:
                    # Use single threshold for all classes
                    pred = pred[pred[:, 4] >= conf_threshold]
                
                if gt.shape[0] == 0:  # No ground truth labels
                    if pred.shape[0] > 0:
                        for pc in pred[:, 5].int():
                            matrix[ti, pc, self.num_classes] += 1  # background FP
                    continue

                if pred.shape[0] == 0:  # No predictions
                    for gc in gt[:, -1].int():
                        matrix[ti, self.num_classes, gc] += 1  # background FN
                    continue

                # Calculate IoU between predictions and ground truths
                iou = box_iou(pred[:, :4], gt[:, :4])
                
                # Find matches above IoU threshold
                matches = torch.nonzero(iou >= iou_threshold)
                if matches.shape[0] > 0:
                    if matches.shape[0] > 1:
                        # Add IoU values to matches
                        matches = torch.cat((matches, iou[matches[:, 0], matches[:, 1]].unsqueeze(1)), 1)
                        
                        # Sort matches by IoU score
                        matches = matches[matches[:, 2].argsort(descending=True)]
                        
                        # Remove duplicate predictions and ground truths
                        unique_pred_idx = torch.unique(matches[:, 1].long(), return_inverse=True)[1]
                        matches = matches[unique_pred_idx]
                        
                        unique_gt_idx = torch.unique(matches[:, 0].long(), return_inverse=True)[1]
                        matches = matches[unique_gt_idx]

                        # Convert matches back to the correct type if needed
                        matches = matches.long()

                    # Get predicted and ground truth classes for matched boxes
                    pred_cls = pred[matches[:, 0], 5].int()
                    gt_cls = gt[matches[:, 1], -1].int()
                    
                    # Process each prediction-ground truth pair
                    for pc, gc in zip(pred_cls, gt_cls):
                        if pc == gc:  # Correct classification
                            matrix[ti, pc, gc] += 1
                        else:  # Misclassification
                            matrix[ti, pc, gc] += 1  # Add to confusion matrix at predicted vs actual position

                    # Add remaining unmatched predictions as false positives
                    unmatched_preds = set(range(pred.shape[0])) - set(matches[:, 0].cpu().numpy())
                    for idx in unmatched_preds:
                        pc = pred[idx, 5].int()
                        matrix[ti, pc, self.num_classes] += 1  # false positive
                    
                    # Add remaining unmatched ground truths as false negatives
                    unmatched_gts = set(range(gt.shape[0])) - set(matches[:, 1].cpu().numpy())
                    for idx in unmatched_gts:
                        gc = gt[idx, -1].int()
                        matrix[ti, self.num_classes, gc] += 1  # false negative
                else:
                    # No matches - all predictions are FP and all ground truths are FN
                    for pc in pred[:, 5].int():
                        matrix[ti, pc, self.num_classes] += 1
                    for gc in gt[:, -1].int():
                        matrix[ti, self.num_classes, gc] += 1

        if plot:
            # The following code is adapted from
            # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py.
            import seaborn

            tick_labels = np.array(["Background"] + [self.idx_to_name[i] for i in sorted(list(self.idx_to_name.keys()))])
            save_dir = os.path.join(self.save_dir,  f"{prefix}_confusionMatrix_IOU={self.iou_threshold[0]}.png")
            array = matrix[0].cpu().detach().numpy().astype(np.float64)
            array /= (array.sum(0).reshape(1, -1) + 1e-9)
            array[array < 0.005] = np.nan

            fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
            nc, nn = self.num_classes, len(self.idx_to_name)  # number of classes, names
            seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size

            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".3f",
                square=True,
                vmin=0.0,
                xticklabels=tick_labels,
                yticklabels=tick_labels,
            ).set_facecolor((1, 1, 1))
            title = "Confusion Matrix"
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title(title)
            fig.savefig(save_dir, dpi=250)
            plt.close(fig)

        return matrix

    def ap_per_class(self, f1=False, plot=False, plot_all=False, prefix=""):
        """
        Calculates AP per class for each IOU threshold given.
        Adapted from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py.

        Args:
            plot (bool): Flag indicating whether to save plots or not.
            plot_all (bool): Indicates whether to save plots for all IOU thresholds.

        Returns:
            ap (Tensor[t, num_classes]): AP for each class for each IOU threshold given.
        """
        # Check if we have valid predictions
        if self.pred_true.shape[1] == 0:
            print("Warning: No valid detections found. Cannot calculate AP.")
            # Return zeros for AP
            ap = torch.zeros((self.iou_threshold.shape[0], len(self.idx_to_name.keys())), 
                            dtype=torch.float64, device=self.device)
            return ap
            
        # turn the tensors into numpy arrays
        detection_true = self.pred_true.numpy(force=True)
        conf = self.conf.numpy(force=True)
        detection_cls = self.pred_cls.numpy(force=True)
        gt_cls = self.gt_cls.numpy(force=True)

        t = self.iou_threshold.shape[0]
        n = detection_true[0].shape[0]
        eps = 1e-7  # to prevent division by 0

        # Sort in order of decreasing conf.
        i = np.argsort(-conf)
        detection_true, conf, detection_cls = detection_true[:, i], conf[i], detection_cls[i]

        # Get unique classes for which labels exist, and number of labels per class
        classes, num_labels = np.unique(gt_cls, return_counts=True)
        nc = classes.shape[0]  # number of classes
        
        if nc == 0:
            print("Warning: No classes found in ground truth labels. Cannot calculate AP.")
            # Return zeros for AP
            ap = torch.zeros((t, len(self.idx_to_name.keys())), 
                           dtype=torch.float64, device=self.device)
            return ap

        # Confidence values at which to calculate precision and recall curves
        x = np.linspace(0, 1, 1000)

        # Average precision, precision and recall curves
        ap = np.zeros((t, nc))

        # p_curve, r_curve: (t, nc, 1000)
        p_curve, r_curve = np.zeros((t, nc, 1000)), np.zeros((t, nc, 1000))

        # prec_values: list giving pr curve for each class
        prec_values = [[] for _ in range(t)]

        for ci, c in enumerate(classes):
            # get indices of detections for this class
            i = detection_cls == c

            n_l = num_labels[ci]  # number of labels
            n_p = np.sum(i)  # number of predictions

            if n_p == 0 or n_l == 0:
                continue

            # calculate FP and TP counts each with shape (t, n_p)
            fp_count = (1 - detection_true[:, i]).cumsum(-1)
            tp_count = detection_true[:, i].cumsum(-1)

            # Recall curve
            recall = tp_count / (n_l + eps)  # recall curve (t, n_p)
            for ti in range(t):
                # -x, -conf because xs must be increasing.
                # also for loop because i believe interp needs 1D arrays.
                r_curve[ti, ci] = np.interp(-x, -conf[i], recall[ti], left=0)

            # Precision curve
            precision = tp_count / (tp_count + fp_count)  # precision curve (t, n_p)
            for ti in range(t):
                p_curve[ti, ci] = np.interp(-x, -conf[i], precision[ti], left=1)

            # Precision-recall curve
            for ti in range(t):  # for each iou threshold
                ap[ti, ci], mpre, mrec = compute_ap(recall[ti], precision[ti])
                prec_values[ti].append(np.interp(x, mrec, mpre))  # precision against recall at the given IOU threshold

        # convert precision-recall curves at each IOU threshold to np arrays
        for ti in range(t):
            prec_values[ti] = np.array(prec_values[ti])  # (nc, 1000)

        # F1 curve
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

        # set average precision
        self.ap = torch.as_tensor(ap, dtype=torch.float64, device=self.device)

        if plot:
            if plot_all:
                iou_thresholds = range(t)
            else:
                # only output graphs for iou threshold = 0.5
                iou_thresholds = [0]

            for ti in iou_thresholds:
                iou = np.round(self.iou_threshold[ti].item(), decimals=2)

                plot_curve(x, p_curve[ti], os.path.join(self.save_dir, f"{prefix}_P_curve_IOU={iou}.png"),
                           self.idx_to_name, xlabel="Confidence", ylabel="Precision")
                plot_curve(x, r_curve[ti], os.path.join(self.save_dir, f"{prefix}_R_curve_IOU={iou}.png"),
                           self.idx_to_name, xlabel="Confidence", ylabel="Recall")
                plot_curve(x, f1_curve[ti], os.path.join(self.save_dir, f"{prefix}_F1_curve_IOU={iou}.png"),
                           self.idx_to_name, xlabel="Confidence", ylabel="F1", show_max=True)
                plot_pr_curve(x, prec_values[ti], ap[ti], self.iou_threshold[ti],
                              os.path.join(self.save_dir, f"{prefix}_PR_curve_IOU={iou}.png"),
                              self.idx_to_name)
        
        
        if f1:
            i = smooth(f1_curve.mean(0), 0.1).argmax()
            return ap,  f1_curve[:, i]

        return ap

    def get_map50(self):
        if self.ap is None:
            self.ap_per_class(plot=False)

        return self.ap[0].mean()

    def get_map50_95(self):
        if self.ap is None:
            self.ap_per_class(plot=False)

        return self.ap.mean(dim=-1).mean()

# Make sure to export the class at the end of the file
__all__ = ['ObjectDetectionMetrics']


