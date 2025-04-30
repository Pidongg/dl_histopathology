from ultralytics import YOLO
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from evaluation import model_utils

def evaluate_with_thresholds(model, data_yaml, iou_thresh, conf_thresholds):
    """Evaluate model with specific thresholds and return metrics."""
    metrics = model.val(
        data=data_yaml,
        conf=min(conf_thresholds),  # Use minimum conf for initial detection
        iou=iou_thresh,
        class_conf_thresholds=conf_thresholds
    )
    
    return {
        'map': metrics.box.map,
        'map50': metrics.box.map50,
        'map75': metrics.box.map75,
        'maps': metrics.box.maps.tolist(),  # Per-class mAP
        'precision': metrics.box.p.tolist(),  # Per-class precision
        'recall': metrics.box.r.tolist(),  # Per-class recall
        'f1': metrics.box.f1.tolist(),  # Per-class F1
    }

def find_optimal_conf_thresholds(model, data_yaml, iou_thresh, class_names, metric='f1', conf_range=None):
    """Find optimal confidence threshold for each class based on the specified metric."""
    best_confs = [0.25] * len(class_names)  # Default starting point
    best_scores = [0] * len(class_names)  # Track best score for each class
    
    # Store dataframes for each class
    all_results_dfs = []
    
    # For each class, find optimal confidence threshold
    for class_idx in range(len(class_names)):
        print(f"Finding optimal threshold for class {class_idx}: {class_names[class_idx]} using {metric} metric")
        
        results_data = {
            'class': [],
            'confidence': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'map50': []
        }
        
        for conf in tqdm(conf_range):
            # Create conf thresholds with current conf for target class
            class_conf_thresholds = [0.25] * len(class_names)
            class_conf_thresholds[class_idx] = conf
            
            # Evaluate with current thresholds
            metrics = evaluate_with_thresholds(model, data_yaml, iou_thresh, class_conf_thresholds)
            
            # Record results
            results_data['class'].append(class_names[class_idx])
            results_data['confidence'].append(conf)
            results_data['f1'].append(metrics['f1'][class_idx])
            results_data['precision'].append(metrics['precision'][class_idx])
            results_data['recall'].append(metrics['recall'][class_idx])
            results_data['map50'].append(metrics['maps'][class_idx])
            
            # Update best threshold if this class's chosen metric score is better
            current_score = model_utils.extract_class_metric(metrics, class_idx, metric)
            if current_score > best_scores[class_idx]:
                best_scores[class_idx] = current_score
                best_confs[class_idx] = conf
        
        # Create DataFrame for this class and add to collection
        class_df = pd.DataFrame(results_data)
        all_results_dfs.append(class_df)
    
    # Combine all class results
    results_df = pd.concat(all_results_dfs, ignore_index=True)
    
    return best_confs, results_df

def find_optimal_iou_threshold(model, data_yaml, conf_thresholds, metric='f1', iou_range=None):
    """Find optimal IoU threshold using optimized confidence thresholds based on average of the specified metric."""
    if iou_range is None:
        iou_range = np.arange(0.3, 0.71, 0.05)  # Default if not specified
    
    best_iou = 0.5  # Default
    best_avg_score = 0
    
    results_data = {
        'iou': [],
        'avg_f1': [],
        'avg_precision': [],
        'avg_recall': [],
        'avg_map50': [],
        'map': [],
        'map50': [],
        'map75': []
    }
    
    print(f"Finding optimal IoU threshold with optimized confidence thresholds based on {metric}")
    
    for iou in tqdm(iou_range):
        # Evaluate with current IoU threshold and optimized conf thresholds
        metrics_data = evaluate_with_thresholds(model, data_yaml, iou, conf_thresholds)
        
        # Calculate average scores across all classes
        avg_f1 = sum(metrics_data['f1']) / len(metrics_data['f1'])
        avg_precision = sum(metrics_data['precision']) / len(metrics_data['precision'])
        avg_recall = sum(metrics_data['recall']) / len(metrics_data['recall'])
        avg_map50 = sum(metrics_data['maps']) / len(metrics_data['maps'])
        
        # Record results
        results_data['iou'].append(iou)
        results_data['avg_f1'].append(avg_f1)
        results_data['avg_precision'].append(avg_precision)
        results_data['avg_recall'].append(avg_recall)
        results_data['avg_map50'].append(avg_map50)
        results_data['map'].append(metrics_data['map'])
        results_data['map50'].append(metrics_data['map50'])
        results_data['map75'].append(metrics_data['map75'])
        
        # Calculate the metric we're optimizing for
        if metric == 'f1':
            avg_score = avg_f1
        elif metric == 'precision':
            avg_score = avg_precision
        elif metric == 'recall':
            avg_score = avg_recall
        elif metric == 'map50':
            avg_score = avg_map50
        elif metric == 'map':
            avg_score = metrics_data['map']
        elif metric == 'map75':
            avg_score = metrics_data['map75']
        
        # Update best threshold if average score is better
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_iou = iou
    
    # Create DataFrame for easier analysis and visualization
    results_df = pd.DataFrame(results_data)
    
    return best_iou, results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to .pt file",
                        default='/local/scratch/pz286/dl_histopathology/weights/yolo11n.pt')
    parser.add_argument("-cfg",
                        help="Path to data config file",
                        default='/local/scratch/pz286/dl_histopathology/config/tau_data_test.yaml')
    parser.add_argument("--output_dir",
                        help="Directory to save results",
                        default='threshold_optimization_results')
    parser.add_argument("--conf_min",
                        type=float,
                        default=0.05,
                        help="Minimum confidence threshold to test")
    parser.add_argument("--conf_max",
                        type=float,
                        default=0.95,
                        help="Maximum confidence threshold to test")
    parser.add_argument("--conf_step",
                        type=float,
                        default=0.1,
                        help="Step size for confidence threshold testing")
    parser.add_argument("--iou_min",
                        type=float,
                        default=0.3,
                        help="Minimum IoU threshold to test")
    parser.add_argument("--iou_max",
                        type=float,
                        default=0.7,
                        help="Maximum IoU threshold to test")
    parser.add_argument("--iou_step",
                        type=float,
                        default=0.05,
                        help="Step size for IoU threshold testing")
    parser.add_argument("--default_iou",
                        type=float,
                        default=0.5,
                        help="Default IoU threshold to use for confidence optimization")
    parser.add_argument("--metric",
                        type=str,
                        choices=['f1', 'precision', 'recall', 'map', 'map50', 'map75'],
                        default='f1',
                        help="Metric to optimize for")

    args = parser.parse_args()

    # Load model and config
    model = YOLO(args.pt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load class names
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    print(f"Loaded {len(class_names)} classes")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define threshold ranges using step sizes
    # Add a small epsilon to max value to ensure it's included in the range
    conf_range = np.arange(args.conf_min, args.conf_max + args.conf_step/2, args.conf_step)
    iou_range = np.arange(args.iou_min, args.iou_max + args.iou_step/2, args.iou_step)

    # First, find optimal confidence thresholds using default IoU
    default_iou = args.default_iou
    print(f"Finding optimal confidence thresholds using default IoU={default_iou} based on {args.metric} scores")
    best_confs, conf_results = find_optimal_conf_thresholds(
        model, args.cfg, default_iou, class_names, args.metric, conf_range
    )
    
    # Now find optimal IoU threshold using the optimized confidence thresholds
    print(f"Finding optimal IoU threshold using optimized confidence thresholds based on {args.metric}")
    best_iou, iou_results = find_optimal_iou_threshold(
        model, args.cfg, best_confs, args.metric, iou_range
    )
    
    # Save results using common utility function
    _ = model_utils.save_optimization_results(
        best_iou, best_confs, class_names, args.metric, 
        conf_results, iou_results, args.output_dir
    )
    
    # Print optimal thresholds
    print(f"Optimal IoU threshold: {best_iou}")
    print("Optimal confidence thresholds:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {best_confs[i]}")
    
    # Final evaluation with optimal thresholds
    print("Running final evaluation with optimal thresholds")
    final_metrics = evaluate_with_thresholds(model, args.cfg, best_iou, best_confs)
    
    # Print metrics per class
    for metric_name in ['f1', 'precision', 'recall']:
        print(f"Final per-class {metric_name} scores:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {final_metrics[metric_name][i]:.4f}")

if __name__ == "__main__":
    main()