import time
from data_preparation import data_utils
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
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
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 0, 4096
    # time_limit = 10.0  # seconds to quit after

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
        x_all = x_all[((x_all[:, 0, 2:4] > min_wh) & (
            x_all[:, 0, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        if not x_all.shape[0]:
            continue

        # No need to compute conf since we don't have obj_conf anymore
        x_all_orig_shape = x_all.shape
        x_all = x_all.reshape(-1, x_all.shape[2])
        x_all = x_all.reshape(x_all_orig_shape)

        if get_unknowns:
            # Getting bboxes only when all the labels are predicted with prob below 0.5
            x_all = x_all[(x_all[:, 0, 4:] < 0.5).all(1) &
                          (x_all[:, 0, 4:] > 0.1).any(1)]
            # x_all = x_all[(x_all[:, 0, 5:] < 0.5).all(1)]
            # x_all = x_all[(x_all[:, 0, 5:] < 0.2).all(1) & (x_all[:, 0, 5:] > 0.1).any(1)]
            if x_all.shape[0] == 0:
                continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        if not use_xyxy_format:
            box = xywh2xyxy(x_all[:, 0, :4])
        else:
            # If already in xyxy format, just use as is
            box = x_all[:, 0, :4]

        # Removing bboxes out of image limits
        wrong_bboxes = (box < 0).any(1) | (box[:, [0, 2]] >= max_width).any(
            1) | (box[:, [1, 3]] >= max_height).any(1)
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
            x = torch.cat(
                (box[i], x_all[i, 0, j + 4].unsqueeze(1), j.float().unsqueeze(1)), 1)
            x_all = x_all[i, ...]
        else:  # best class only
            conf, j = x_all[:, 0, 4:].max(1)

            if class_conf_thresholds is not None:
                # Check if each detection's best class passes its specific threshold
                class_thresholds = torch.tensor(
                    class_conf_thresholds, device=conf.device)
                # Get threshold for each detection based on its class
                thresholds = class_thresholds[j.long()]
                # Filter based on class-specific thresholds
                mask = conf > thresholds
            else:
                # Use global threshold
                mask = conf > conf_thres

            x = torch.cat((box, conf.unsqueeze(
                1), j.float().unsqueeze(1)), 1)[mask]
            x_all = x_all[mask, ...]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * \
            max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

        output[xi] = x[i]

        x_all = x_all[i, :, :]
        all_scores[xi] = x_all[:, 0, 4:]

        if prediction.shape[2] > 1:
            sampled_coords[xi] = x_all[:, 1:, :4]

    return output, all_scores, sampled_coords


def visualize_conf_thresholds(results_df, metric='f1', save_path='conf_threshold_optimization.png', single_plot=False):
    """
    Visualize confidence threshold optimization results.

    Args:
        results_df: DataFrame with optimization results
        metric: Metric used for optimization (f1, precision, recall, map50, etc.)
        save_path: Path to save the visualization
        single_plot: If True, only create a single plot for the specified metric
    """
    # Map metric to display name
    metric_labels = {
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'map50': 'mAP50',
        'map': 'mAP'
    }
    metric_label = metric_labels.get(metric, metric.upper())

    # Create a color palette
    palette = sns.color_palette(
        "husl", n_colors=len(results_df['class'].unique()))

    if single_plot:
        # Single plot for the specified metric
        plt.figure(figsize=(10, 6))

        for i, class_name in enumerate(results_df['class'].unique()):
            class_data = results_df[results_df['class'] == class_name]
            plt.plot(class_data['confidence'], class_data[metric],
                     marker='o', label=class_name, color=palette[i])

        plt.xlabel('Confidence Threshold')
        plt.ylabel(metric_label)
        plt.title(f'{metric_label} vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Create multi-plot figure if all metrics are available
        available_metrics = [col for col in [
            'f1', 'precision', 'recall', 'map50'] if col in results_df.columns]
        n_plots = len(available_metrics)

        if n_plots <= 2:
            fig_height, fig_width = 6, 12
        else:
            fig_height, fig_width = 10, 15

        plt.figure(figsize=(fig_width, fig_height))

        for plot_idx, plot_metric in enumerate(available_metrics, 1):
            plt.subplot(2, (n_plots+1)//2, plot_idx)

            for i, class_name in enumerate(results_df['class'].unique()):
                class_data = results_df[results_df['class'] == class_name]
                plt.plot(class_data['confidence'], class_data[plot_metric], marker='o',
                         label=class_name, color=palette[i])

            plot_metric_label = metric_labels.get(
                plot_metric, plot_metric.upper())
            plt.xlabel('Confidence Threshold')
            plt.ylabel(plot_metric_label)
            plt.title(f'{plot_metric_label} vs Confidence Threshold')
            plt.grid(True, alpha=0.3)

            # Only add legend to the first subplot
            if plot_idx == 1:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_iou_thresholds(results_df, metric='f1', save_path='iou_threshold_optimization.png'):
    """
    Visualize IoU threshold optimization results.

    Args:
        results_df: DataFrame with optimization results
        metric: Metric used for optimization (f1, precision, recall, map50, etc.)
        save_path: Path to save the visualization
    """
    # Map metric to display name
    metric_labels = {
        'f1': 'Average F1 Score',
        'precision': 'Average Precision',
        'recall': 'Average Recall',
        'map50': 'Average mAP50',
        'map': 'mAP (0.5-0.95)',
    }
    metric_label = metric_labels.get(metric, metric.upper())

    # Determine which columns are available in the results dataframe
    available_metrics = []
    metric_columns = {}

    # Map common metrics to their column names
    for m in ['f1', 'precision', 'recall', 'map50', 'map']:
        avg_col = f'avg_{m}'
        if avg_col in results_df.columns:
            available_metrics.append(m)
            metric_columns[m] = avg_col
        elif m in results_df.columns:
            available_metrics.append(m)
            metric_columns[m] = m

    # Create visualization based on available metrics
    if len(available_metrics) >= 4:
        # Use 2x2 grid if we have 4+ metrics
        plt.figure(figsize=(12, 10))

        # Plot the optimization metric first
        plt.subplot(2, 2, 1)
        y_column = metric_columns.get(metric, metric)
        plt.plot(results_df['iou'], results_df[y_column],
                 marker='o', linewidth=2, color='green')
        plt.xlabel('IoU Threshold')
        plt.ylabel(metric_label)
        plt.title(f'{metric_label} vs IoU Threshold')
        plt.grid(True, alpha=0.3)

        # Plot other important metrics in remaining slots
        plot_idx = 2
        for m in ['map', 'map50', 'f1', 'precision', 'recall']:
            if m != metric and m in available_metrics and plot_idx <= 4:
                plt.subplot(2, 2, plot_idx)
                y_column = metric_columns.get(m, m)
                plt.plot(
                    results_df['iou'], results_df[y_column], marker='o', linewidth=2)
                plt.xlabel('IoU Threshold')
                plt.ylabel(metric_labels.get(m, m.upper()))
                plt.title(
                    f'{metric_labels.get(m, m.upper())} vs IoU Threshold')
                plt.grid(True, alpha=0.3)
                plot_idx += 1
    else:
        plt.figure(figsize=(12, 5))

        if 'map' in available_metrics:
            plt.subplot(1, 2, 1)
            plt.plot(
                results_df['iou'], results_df[metric_columns['map']], marker='o', linewidth=2)
            plt.xlabel('IoU Threshold')
            plt.ylabel('mAP (0.5-0.95)')
            plt.title('mAP vs IoU Threshold')
            plt.grid(True, alpha=0.3)
        elif 'map50' in available_metrics:
            plt.subplot(1, 2, 1)
            plt.plot(
                results_df['iou'], results_df[metric_columns['map50']], marker='o', linewidth=2)
            plt.xlabel('IoU Threshold')
            plt.ylabel('mAP50')
            plt.title('mAP50 vs IoU Threshold')
            plt.grid(True, alpha=0.3)
        else:
            plt.subplot(1, 2, 1)
            y_column = metric_columns.get(metric, metric)
            plt.plot(results_df['iou'], results_df[y_column],
                     marker='o', linewidth=2)
            plt.xlabel('IoU Threshold')
            plt.ylabel(metric_label)
            plt.title(f'{metric_label} vs IoU Threshold')
            plt.grid(True, alpha=0.3)

        # Show optimization metric or another metric on right
        plt.subplot(1, 2, 2)
        if metric != 'map' and metric != 'map50' and metric in available_metrics:
            y_column = metric_columns.get(metric, metric)
            plt.plot(results_df['iou'], results_df[y_column],
                     marker='o', linewidth=2)
            plt.xlabel('IoU Threshold')
            plt.ylabel(metric_label)
            plt.title(f'{metric_label} vs IoU Threshold')
            plt.grid(True, alpha=0.3)
        elif 'f1' in available_metrics and metric != 'f1':
            plt.plot(
                results_df['iou'], results_df[metric_columns['f1']], marker='o', linewidth=2)
            plt.xlabel('IoU Threshold')
            plt.ylabel('F1 Score')
            plt.title('F1 Score vs IoU Threshold')
            plt.grid(True, alpha=0.3)
        elif len(available_metrics) > 1:
            # Use any available metric that's not already plotted
            for m in available_metrics:
                if m != metric and (m != 'map' and m != 'map50' or (metric == 'map' or metric == 'map50')):
                    y_column = metric_columns.get(m, m)
                    plt.plot(
                        results_df['iou'], results_df[y_column], marker='o', linewidth=2)
                    plt.xlabel('IoU Threshold')
                    plt.ylabel(metric_labels.get(m, m.upper()))
                    plt.title(
                        f'{metric_labels.get(m, m.upper())} vs IoU Threshold')
                    plt.grid(True, alpha=0.3)
                    break

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_optimization_results(best_iou, best_confs, class_names, metric, conf_results, iou_results, output_dir):
    """
    Save threshold optimization results to files.

    Args:
        best_iou: Best IoU threshold
        best_confs: List of best confidence thresholds per class
        class_names: List of class names
        metric: Metric used for optimization
        conf_results: Confidence optimization results dataframe
        iou_results: IoU optimization results dataframe
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save confidence results
    conf_results.to_csv(os.path.join(
        output_dir, 'conf_optimization_results.csv'), index=False)

    # Visualize confidence results
    conf_vis_path = os.path.join(output_dir, 'conf_threshold_optimization.png')
    visualize_conf_thresholds(conf_results, metric, conf_vis_path)

    # Save IoU results
    iou_results.to_csv(os.path.join(
        output_dir, 'iou_optimization_results.csv'), index=False)

    # Visualize IoU results
    iou_vis_path = os.path.join(output_dir, 'iou_threshold_optimization.png')
    visualize_iou_thresholds(iou_results, metric, iou_vis_path)

    # Save optimal thresholds to JSON
    optimal_thresholds = {
        'iou_threshold': float(best_iou),
        'confidence_thresholds': {class_names[i]: float(best_confs[i]) for i in range(len(class_names))},
        'optimization_metric': metric
    }

    with open(os.path.join(output_dir, 'optimal_thresholds.json'), 'w') as f:
        json.dump(optimal_thresholds, f, indent=4)

    # Return the paths to results for reference
    return {
        'conf_csv': os.path.join(output_dir, 'conf_optimization_results.csv'),
        'conf_plot': conf_vis_path,
        'iou_csv': os.path.join(output_dir, 'iou_optimization_results.csv'),
        'iou_plot': iou_vis_path,
        'thresholds_json': os.path.join(output_dir, 'optimal_thresholds.json')
    }


def process_metrics(metrics, metric_name='f1'):
    """
    Extract and process metrics values, handling different formats that might come from different models.

    Args:
        metrics: Metrics object or dictionary from model evaluation
        metric_name: Name of the metric to extract

    Returns:
        Processed metric value (float)
    """
    if metric_name not in metrics:
        raise ValueError(f"Metric {metric_name} not found in metrics")

    metric_value = metrics[metric_name]

    if isinstance(metric_value, torch.Tensor):
        metric_value = metric_value.tolist()

    return metric_value


def extract_class_metric(metrics, class_idx, metric_name='f1'):
    """
    Extract a specific class's metric value from metrics.

    Args:
        metrics: Metrics dictionary from evaluation
        class_idx: Class index to extract
        metric_name: Name of the metric to extract

    Returns:
        Class-specific metric value (float)
    """
    try:
        metric_values = metrics[metric_name]

        if isinstance(metric_values, torch.Tensor):
            metric_values = metric_values.tolist()

        class_value = metric_values[class_idx]

        # Handle nested lists
        if isinstance(class_value, list):
            class_value = class_value[0]

        # Handle tensors
        if isinstance(class_value, torch.Tensor):
            class_value = class_value.item()

        return float(class_value)
    except (IndexError, KeyError, TypeError, AttributeError):
        return 0.0


def monte_carlo_predictions(self, img, conf_thresh, iou_thresh, model_type, input_size, num_samples=30):
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
            if model_type == 'rcnn':
                if isinstance(predictions, list):
                    predictions = predictions[0]
            else:
                predictions = predictions[:, :8, :]
                predictions = predictions.transpose(-2, -1)

            infs_all.append(predictions.unsqueeze(2))

        if num_samples > 1:
            inf_mean = torch.mean(torch.stack(infs_all), dim=0)
            infs_all.insert(0, inf_mean)
            inf_out = torch.cat(infs_all, dim=2)
        else:
            inf_out = infs_all[0]

        model_height, model_width = input_size, input_size
        # Apply NMS and get sampled coordinates
        output, all_scores, sampled_coords = non_max_suppression(
            inf_out,
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            multi_label=False,
            max_width=model_width,
            max_height=model_height
        )
        image_width = 512  # Images are 512x512 in our tau histopathology dataset
        scale_factor = image_width / model_width

        # Process sampled coordinates if we have multiple samples
        if num_samples > 1 and output[0] is not None and len(output[0]) > 0:
            output[0][:, :4] *= scale_factor

            # Process sampled coordinates for the first (and only) image in batch
            sampled_boxes = data_utils.xywh2xyxy(
                sampled_coords[0].reshape(-1, 4)).reshape(sampled_coords[0].shape)
            sampled_boxes *= scale_factor
            data_utils.clip_coords(
                sampled_boxes.reshape(-1, 4), (model_height, model_width))

            # Calculate covariances for each detection
            covariances = torch.zeros(
                sampled_boxes.shape[0], 2, 2, 2, device=output[0].device)
            for det_idx in range(sampled_boxes.shape[0]):
                covariances[det_idx, 0, ...] = data_utils.cov(
                    sampled_boxes[det_idx, :, :2])  # covariance of top left corner
                covariances[det_idx, 1, ...] = data_utils.cov(
                    sampled_boxes[det_idx, :, 2:])  # covariance of bottom right corner

            # Ensure covariances are positive semi-definite
            for det_idx in range(len(covariances)):
                for i in range(2):
                    cov_matrix = covariances[det_idx, i].cpu().numpy()
                    if not data_utils.is_pos_semidef(cov_matrix):
                        print('Converting covariance matrix to near PSD')
                        cov_matrix = data_utils.get_near_psd(cov_matrix)
                        covariances[det_idx, i] = torch.tensor(
                            cov_matrix, device=output[0].device)

            # Round values for smaller size
            covariances = torch.round(covariances * 1e6) / 1e6
        elif output[0] is not None and len(output[0]) > 0:
            # If we only have one sample but still have valid output, apply scaling
            output[0][:, :4] *= scale_factor
            covariances = None
        else:
            covariances = None
    return output, all_scores, covariances
