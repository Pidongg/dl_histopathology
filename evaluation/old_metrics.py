# Computing validation metrics for models.

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


def smooth(y, f=0.05):
    """
    Box filter of fraction f.

    Implementation from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py
    """
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def plot_curve(px, py, save_dir, names: dict[int, str], xlabel, ylabel):
    """
    Plots a curve.
    Implementation from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    for i, y in enumerate(py):
        ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-{xlabel} Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=(), on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

class ObjectDetectionMetrics:
    """
    This class is for computing detection metrics such as mean average precision (mAP) of an object detection model.
    Based on https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        idx_to_name (Dict[int, str]): A dictionary that maps class indexes to names.
        num_classes (int): Number of classes.
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
    def __init__(self, save_dir, idx_to_name, num_classes: int, detections: list[Tensor], ground_truths: list[Tensor], device):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.idx_to_name = idx_to_name
        self.num_classes = num_classes
        self.device = device

        self.detections = detections
        self.ground_truths = ground_truths

        self.num_detections = 0
        self.num_labels = 0
        for detection in detections:
            self.num_detections += detection.shape[0]

        for gt in ground_truths:
            self.num_labels += gt.shape[0]

    def get_confusion_matrix(self, iou_threshold, conf_threshold):
        """
        Returns a confusion matrix with predicted class as row index, and gt class as col index.
        """
        matrix = torch.zeros((self.num_classes + 1, self.num_classes + 1))  # confusion matrix

        detection_true, conf, detection_cls, gt_cls, gt_per_pred_all, pred_per_gt_all = \
            self.match_predictions(Tensor([iou_threshold]))

        # # -------- DELETE LATER -------------
        # print("detection classes: ", detection_cls[:15])
        # print("gt classes: ", gt_cls[:15])
        # print("gt per pred: ", gt_per_pred_all[:15])
        # print("pred per gt: ", pred_per_gt_all[:15])
        # # -----------------------------------

        # only one iou threshold used, so take out the corresponding tensors
        gt_per_pred_all = gt_per_pred_all[0]
        pred_per_gt_all = pred_per_gt_all[0]

        # filter by confidence
        # first update gt_per_pred_all to remove the unused predictions
        gt_per_pred_all = gt_per_pred_all[conf >= conf_threshold]

        # then update pred_per_gt_all to remove invalid predictions
        pred_per_gt_all[conf[pred_per_gt_all] < conf_threshold] = -1

        # filter detection classes
        detection_cls = detection_cls[conf >= conf_threshold]

        # objects wrongly predicted as background:
        # find labels that don't correspond to any preds, i.e. do not appear in gt_per_pred.
        # pred_per_gt = torch.zeros(gt_cls.shape[0]) - 1
        # pred_per_gt[gt_pred_matches.nonzero()[:, 1]] = gt_pred_matches.nonzero()[:, 0].float()

        unmatched_labels_idx = torch.where(pred_per_gt_all == -1)[0]

        # note that there is no matrix[0, 0] as there is an infinite number of background boxes...
        for ci in range(self.num_classes):
            num_unpredicted = sum(gt_cls[unmatched_labels_idx] == ci)
            matrix[0, ci + 1] = num_unpredicted

        # get the indices of the labels to which each prediction in this class corresponds.
        # gt_per_pred = torch.zeros(detection_cls.shape[0]) - 1
        # gt_per_pred[gt_pred_matches.nonzero()[:, 0]] = gt_pred_matches.nonzero()[:, 1].float()

        for ci in range(self.num_classes):
            i = detection_cls == ci

            matched_labels_idx = gt_per_pred_all[i].int()

            # matched_classes: the ground truth class of each prediction.
            matched_classes = gt_cls[matched_labels_idx].detach() + 1  # increment to allow for background class.
            matched_classes[matched_labels_idx == -1] = 0  # background class, i.e. no gt matches.

            # print(ci, matched_classes)

            for cj in range(self.num_classes + 1):
                matrix[ci + 1, cj] = sum(matched_classes == cj)

        return matrix


    def match_predictions(self, iou_threshold: Tensor):
        """
        Matches `self.detections` to `self.ground_truths` for each image and for each iou_threshold specified.

        Args:
            iou_threshold: Tensor[t] of IOU thresholds to accept a detection.

        Returns:
            detection_true (Tensor[t, n]): Binary values indicating whether each detection is correct (1) or false (0),
                for each value in iou_threshold.
            conf (Tensor[n]): Confidence for each detection.
            detection_cls (Tensor[n]): Class of each detection.
            gt_cls (Tensor[m]): Class of each ground truth label.
            gt_per_pred (Tensor[n]): Holds the value of the gt matching each pred, and -1 for no matches.
            pred_per_gt (Tensor[n]): Holds the value of the pred matching each gt, and -1 for no matches.
        """
        t = iou_threshold.shape[0]
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
            iou = box_iou(pred[:, :4], gt[:, :4])

            # calculate pred_true for each prediction.

            # tensor[t, num_preds]
            correct = np.zeros((t, pred_cls.shape[0]))

            # LxD matrix where L - labels, D - detections but todo: i really want to transpose this bc wtf
            correct_class = gt_cls[:, None] == pred_cls
            iou = iou.T
            iou = iou * correct_class  # zero out the wrong classes
            iou = iou.cpu().numpy()
            # pred_per_gt: tensor[t, num_gt]
            pred_per_gt = np.zeros((t, m)) - 1

            # gt_per_pred: tensor[t, num_pred]
            gt_per_pred = np.zeros((t, n)) - 1

            for i, threshold in enumerate(iou_threshold.cpu().tolist()):
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T

                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        # store the (gt, pred) pairs in descending order of iou
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                        pred_per_gt[i, matches[:, 0]] = matches[:, 1] + num_preds_curr
                        gt_per_pred[i, matches[:, 1]] = matches[:, 0] + num_gt_curr

                    correct[i, matches[:, 1].astype(int)] = 1
            pred_true_all.append(torch.tensor(correct, dtype=torch.int, device=self.device))

            pred_per_gt_all.append(torch.tensor(pred_per_gt, dtype=torch.int, device=self.device))
            num_preds_curr += n
            gt_per_pred_all.append(torch.tensor(gt_per_pred, dtype=torch.int, device=self.device))
            num_gt_curr += m

            # # !!!!!!!!!!!!!!!!!!
            # # keep only the maximum iou value for each pred, so that each pred matches at most 1 label.
            # # _, max_i = torch.max(iou, axis=-1)
            # # new_iou = torch.zeros(iou.shape)
            # # new_iou[range(n), max_i] = iou[range(n), max_i]
            # # iou = new_iou
            # # print(iou, detection[:, -1], gt[:, -1])
            #
            # iou = torch.stack([iou for _ in range(t)], axis=0).to(self.device)  # shape [t, n, m]
            #
            # # get the indices of boxes that match based on each iou threshold.
            # # idx will be a triple of tensors, with each tensor representing one dimension.
            # # e.g. (idx[0][0], idx[1][0], idx[2][0] specifies the index of the first match.
            # idx = torch.where(iou > iou_threshold.unsqueeze(-1).unsqueeze(-1).to(self.device))
            #
            # # all_matches: tensor[t, num_pred, num_gt].
            # # all_matches[i, j, k] = 1 if detection[j] is matched to gt[k] with iou > i.
            # all_matches = torch.zeros((t, pred.shape[0], gt.shape[0])).to(self.device)
            # all_matches[idx[0], idx[1], idx[2]] = 1
            #
            # # indices of matches
            # match_idx = all_matches.nonzero()
            #
            # # pred_per_gt: tensor[t, num_gt]
            # pred_per_gt = torch.zeros((t, m)) - 1
            # pred_per_gt[match_idx[:, 0], match_idx[:, 2]] = match_idx[:, 1].float() + num_preds_curr
            # pred_per_gt_all.append(pred_per_gt.int())
            # num_preds_curr += n
            #
            # # gt_per_pred: tensor[t, num_pred]
            # gt_per_pred = torch.zeros((t, n)) - 1
            # gt_per_pred[match_idx[:, 0], match_idx[:, 1]] = match_idx[:, 2].float() + num_gt_curr
            # gt_per_pred_all.append(gt_per_pred.int())
            # num_gt_curr += m
            #
            # # ------------- delete later -------------------
            # # di = torch.where(gt[gt_per_pred[0].int()][:, -1] == detection[:, -1])[0]
            # # # print(di)
            # # matched_labels = gt[di, -1]
            # # # print(matched_labels)
            # # print(matched_labels[matched_labels != 4])
            # # -------------------
            #
            # # matches: tensor[t, num_pred, num_gt].
            # # matches[i, j, k] = 1 if detection[j] has the same class as gt[k] with iou > i.
            # matches = torch.zeros((t, pred.shape[0], gt.shape[0])).to(self.device)
            # matches[idx[0], idx[1], idx[2]] = (pred[idx[1], -1] == gt[idx[2], -1]).float()
            #
            # # correctness: tensor[t, num_pred].
            # # correctness[i, j] = 1 if detection[j] is correct with iou > i.
            # # correctness, _ = torch.max(matches, axis=-1)
            # # .append(correctness)
            #
            # # --------------- DELETE LATER ------------------------
            # # print("detection: ", detection)
            # # print("ground truths: ", gt)
            # # print("iou: ", iou)
            # # print("detection classes: ", detection[:, -1])
            # # print("gt per pred: ", end="")
            # # gt_per_pred = gt_per_pred[0] - (num_gt_curr - m)
            # # temp = torch.zeros(gt_per_pred.shape[0]) - 1
            # # temp[gt_per_pred != -num_gt_curr + m - 1] = gt[gt_per_pred[gt_per_pred != -num_gt_curr + m - 1].int()][:, -1]
            # # print(temp)
            # # print("detection true: ", detection_true[-1])
            # # --------------------------------------

        # Concatenate the return lists into tensors
        pred_true_all = torch.cat(pred_true_all, axis=-1)
        gt_per_pred_all = torch.cat(gt_per_pred_all, axis=-1)

        # ---------- DELETE LATER --------------
        # temp = torch.zeros(gt_per_pred_all.shape) - 1
        # gt = torch.cat(self.ground_truths, axis=0)
        #
        # print(gt_per_pred_all.shape, gt.shape, num_preds_curr)
        # temp[gt_per_pred_all != -1] = gt[gt_per_pred_all[gt_per_pred_all != -1].int()][:, -1]
        # print(temp)
        # ------------------------------------

        pred_per_gt_all = torch.cat(pred_per_gt_all, axis=-1)

        # Concatenate conf and class info
        conf = torch.cat(conf_all, axis=-1)
        pred_cls = torch.cat(pred_cls_all, axis=-1)
        gt_cls = torch.cat(gt_cls_all, axis=-1)

        return pred_true_all, conf, pred_cls, gt_cls, gt_per_pred_all, pred_per_gt_all

    def ultralytics_match_predictions(self, iouv: Tensor):
        """
        Matches `self.detections` to `self.ground_truths` for each image and for each iou_threshold specified.

        Args:
            iou_threshold: Tensor[t] of IOU thresholds to accept a detection.

        Returns:
            detection_true (Tensor[t, n]): Binary values indicating whether each detection is correct (1) or false (0),
                for each value in iou_threshold.
            conf (Tensor[n]): Confidence for each detection.
            detection_cls (Tensor[n]): Class of each detection.
            gt_cls (Tensor[m]): Class of each ground truth label.
            gt_per_pred (Tensor[n]): Holds the value of the gt matching each pred, and -1 for no matches.
            pred_per_gt (Tensor[n]): Holds the value of the pred matching each gt, and -1 for no matches.
        """
        conf = []
        pred_cls = []
        gt_cls = []

        detection_true = []

        for i in range(len(self.detections)):
            pred = self.detections[i]
            truth = self.ground_truths[i]

            pred_classes = pred[:, -1]
            gt_classes = truth[:, -1]
            iou = box_iou(pred[:, :4], truth[:, :4])

            conf.append(pred[:, -2])
            pred_cls.append(pred[:, -1])
            gt_cls.append(truth[:, -1])

            # Dx10 matrix, where D - detections, 10 - IoU thresholds
            correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)

            # LxD matrix where L - labels (rows), D - detections (columns)
            correct_class = gt_classes[:, None] == pred_classes
            iou = iou.T
            iou = iou * correct_class  # zero out the wrong classes
            iou = iou.cpu().numpy()
            for j, threshold in enumerate(iouv.cpu().tolist()):
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T  # array of (gt, pred) index pairs which match
                if matches.shape[0]:  # if at least one match
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]  # store the (gt, pred) pairs in descending order of iou
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # filter to leave only unique pred indices, so one label per pred
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # filter to leave one pred per label
                    correct[matches[:, 1].astype(int), j] = True
            detection_true.append(torch.tensor(correct, dtype=torch.bool, device=pred_classes.device))

        detection_true = torch.cat(detection_true, axis=0)

        return detection_true

    def compute_ap(self, recall, precision):
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

    def ap_per_class(self, iou_threshold: Tensor, plot=False):
        """
        Calculates AP per class for each IOU threshold given.
        Based on https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py.

        Args:
            iou_threshold (Tensor[t]): Tensor[t] of IOU thresholds to accept a detection.
            plot (bool): Flag indicating whether to save plots or not.

        Returns:
            ap (Tensor[t, num_classes]): AP for each class for each IOU threshold given.
        """
        detection_true, conf, detection_cls, gt_cls, gt_per_pred_all, pred_per_gt_all = self.match_predictions(iou_threshold)

        # turn the tensors into numpy arrays
        detection_true = detection_true.numpy(force=True)
        conf = conf.numpy(force=True)
        detection_cls = detection_cls.numpy(force=True)
        gt_cls = gt_cls.numpy(force=True)

        t = iou_threshold.shape[0]
        n = detection_true[0].shape[0]
        eps = 1e-7  # to prevent division by 0

        # Sort in order of decreasing conf.
        i = np.argsort(-conf)
        detection_true, conf, detection_cls = detection_true[:, i], conf[i], detection_cls[i]

        # Get unique classes for which labels exist, and number of labels per class
        classes, num_labels = np.unique(gt_cls, return_counts=True)
        nc = classes.shape[0]  # number of classes

        # Confidence values at which to calculate precision and recall curves
        x = np.linspace(0, 1, 1000)

        # Average precision, precision and recall curves
        # ap: (t, nc, n)
        ap = np.zeros((t, nc, n))

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

            # # Visualisation to check behaviour
            # fig, ax = plt.subplots(1, 2, figsize=(9, 6), tight_layout=True)
            # ax[0].plot(conf[i], tp_count[0])
            # ax[0].set_xlabel("Confidence")
            # ax[0].set_ylabel("True positives")
            #
            # ax[1].plot(conf[i], fp_count[0])
            # ax[1].set_xlabel("Confidence")
            # ax[1].set_ylabel("False positives")
            # plt.show()

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
                ap[ti, ci], mpre, mrec = self.compute_ap(recall[ti], precision[ti])
                prec_values[ti].append(np.interp(x, mrec, mpre))  # precision against recall at the given IOU threshold

        # convert precision-recall curves at each IOU threshold to np arrays
        for ti in range(t):
            prec_values[ti] = np.array(prec_values[ti])  # (nc, 1000)

        # F1 curve
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

        if plot:
            for ti in range(t):
                iou = np.round(iou_threshold[ti].item(), decimals=2)

                plot_curve(x, p_curve[ti], os.path.join(self.save_dir, f"P_curve_IOU={iou}.png"),
                           self.idx_to_name, xlabel="Confidence", ylabel="Precision")
                plot_curve(x, r_curve[ti], os.path.join(self.save_dir, f"R_curve_IOU={iou}.png"),
                           self.idx_to_name, xlabel="Confidence", ylabel="Recall")
                plot_curve(x, f1_curve[ti], os.path.join(self.save_dir, f"F1_curve_IOU={iou}.png"),
                           self.idx_to_name, xlabel="Confidence", ylabel="F1")
                plot_pr_curve(x, prec_values[ti], os.path.join(self.save_dir, f"PR_curve_IOU={iou}.png"),
                           self.idx_to_name, xlabel="Recall", ylabel="Precision")

        return r_curve, p_curve

    def ultralytics_ap_per_class(
            self, tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir="", names=(), eps=1e-16, prefix=""
    ):
        """
        Computes the average precision per class for object detection evaluation.

        Args:
            tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
            conf (np.ndarray): Array of confidence scores of the detections.
            pred_cls (np.ndarray): Array of predicted classes of the detections.
            target_cls (np.ndarray): Array of true classes of the detections.
            plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
            on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
            save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
            names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
            prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

        Returns:
            (tuple): A tuple of six arrays and one array of unique classes, where:
                tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
                fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
                p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
                r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
                f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
                ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
                unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
                p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
                r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
                f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
                x (np.ndarray): X-axis values for the curves. Shape: (1000,).
                prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        x, prec_values = np.linspace(0, 1, 1000), []

        # Average precision, precision and recall curves
        ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        prec_values = np.array(prec_values)  # (nc, 1000)

        # Compute F1 (harmonic mean of precision and recall)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
        names = [names[k] for k in names if k in unique_classes]  # list: only classes that have data
        names = dict(enumerate(names))  # to dict
        if plot:
            plot_pr_curve(x, prec_values, ap, os.path.join(save_dir, f"{prefix}PR_curve.png"), names, on_plot=on_plot)
            plot_curve(x, f1_curve, os.path.join(save_dir, f"{prefix}F1_curve.png"), names, ylabel="F1", on_plot=on_plot)
            plot_curve(x, p_curve, os.path.join(save_dir, f"{prefix}P_curve.png"), names, ylabel="Precision", on_plot=on_plot)
            plot_curve(x, r_curve, os.path.join(save_dir, f"{prefix}R_curve.png"), names, ylabel="Recall", on_plot=on_plot)

        i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def match_predictions(pred_classes, true_classes, iou, iouv, use_scipy=False):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
    """
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)

    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou.T
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
