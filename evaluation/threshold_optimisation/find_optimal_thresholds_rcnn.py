import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import argparse
import yaml
from tqdm import tqdm
from evaluator import RCNNEvaluator
import train_model.rcnn_scripts.run_train_rcnn as run_train_rcnn
from evaluation import model_utils

def evaluate_with_thresholds(checkpoint_path, iou_thresh, conf_thresholds, class_names, test_images, test_labels, device, save_dir):
    """Evaluate model with specific thresholds and return metrics."""
    # Convert list of thresholds to dictionary with 1-indexed class IDs as keys
    class_specific_thresholds = {i+1: thresh for i, thresh in enumerate(conf_thresholds)}
    
    # Create a fresh model with the specified thresholds
    num_classes = len(class_names) + 1  # +1 for background class
    model_with_thresholds = run_train_rcnn.get_model_instance_segmentation(
        num_classes, 
        all_scores=True,
        iou_thresh=iou_thresh,
        conf_thresh=min(conf_thresholds),  # Use minimum conf for initial detection
        class_conf_thresh=class_specific_thresholds
    )
    model_with_thresholds.to(device)
    
    # Load weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_with_thresholds.load_state_dict(checkpoint)
    
    evaluator = RCNNEvaluator(
        model_with_thresholds,
        test_imgs=test_images,
        test_labels=test_labels,
        device=device,
        class_dict=class_names,
        save_dir=save_dir
    )
    
    # Get metrics
    ap, f1 = evaluator.ap_per_class(f1=True)
    map50 = evaluator.map50()
    map50_95 = evaluator.map50_95()
    
    return {
        'map': map50_95,
        'map50': map50,
        'maps': ap.tolist() if hasattr(ap, 'tolist') else ap,  # Per-class mAP
        'f1': f1.tolist() if hasattr(f1, 'tolist') else f1
    }

def find_optimal_conf_thresholds(checkpoint_path, iou_thresh, class_names, test_images, test_labels, device, save_dir, metric='f1', conf_range=None):
    """Find optimal confidence threshold for each class based on the specified metric."""
    if conf_range is None:
        conf_range = np.arange(0.05, 0.96, 0.1)  # Default if not specified
    
    best_confs = [0.25] * len(class_names)  # Default starting point
    best_scores = [0] * len(class_names)  # Track best metric score for each class
    
    # Store dataframes for each class
    all_results_dfs = []
    
    # For each class, find optimal confidence threshold
    for class_idx in range(len(class_names)):
        print(f"Finding optimal threshold for class {class_idx}: {class_names[class_idx]} using {metric} metric")
        
        # Initialize results data for this class
        results_data = {
            'class': [],
            'confidence': [],
            'f1': [],
            'map50': []
        }
        
        for conf in tqdm(conf_range):
            # Create conf thresholds with current conf for target class
            class_conf_thresholds = [0.25] * len(class_names)  # Default for other classes
            class_conf_thresholds[class_idx] = conf
            
            # Evaluate with current thresholds
            metrics = evaluate_with_thresholds(checkpoint_path, iou_thresh, class_conf_thresholds, class_names, test_images, test_labels, device, save_dir)
            
            # Extract metrics safely using our utility function
            per_class_f1 = model_utils.extract_class_metric(metrics, class_idx, 'f1')
            per_class_map50 = model_utils.extract_class_metric(metrics, class_idx, 'maps')
            
            # Record results
            results_data['class'].append(class_names[class_idx])
            results_data['confidence'].append(float(conf))
            results_data['f1'].append(float(per_class_f1))
            results_data['map50'].append(float(per_class_map50))
            
            # Determine which metric value to use for optimization
            if metric == 'f1':
                metric_value = per_class_f1
            elif metric == 'map50':
                metric_value = per_class_map50
            elif metric == 'map':
                # For mAP, we use the overall model mAP, not class-specific
                metric_value = metrics['map'].item() if isinstance(metrics['map'], torch.Tensor) else metrics['map']
            else:
                # Default to F1
                metric_value = per_class_f1
            
            # Update best threshold if this class's metric score is better
            if metric_value > best_scores[class_idx]:
                best_scores[class_idx] = metric_value
                best_confs[class_idx] = conf
        
        # Create DataFrame for this class and add to collection
        class_df = pd.DataFrame(results_data)
        all_results_dfs.append(class_df)
    
    # Combine all class results
    results_df = pd.concat(all_results_dfs, ignore_index=True)
    
    return best_confs, results_df

def find_optimal_iou_threshold(checkpoint_path, conf_thresholds, class_names, test_images, test_labels, device, save_dir, metric='f1', iou_range=None):
    """Find optimal IoU threshold using optimized confidence thresholds based on the specified metric."""
 
    best_iou = 0.5
    best_score = 0
    
    results_data = {
        'iou': [],
        'map': [],
        'map50': [],
        'f1': []
    }
    
    print(f"Finding optimal IoU threshold with optimized confidence thresholds using {metric} metric")
    
    for iou in tqdm(iou_range):
        # Evaluate with current IoU threshold and optimized conf thresholds
        metrics = evaluate_with_thresholds(checkpoint_path, iou, conf_thresholds, class_names, test_images, test_labels, device, save_dir)
        
        # Convert tensor values to Python floats if necessary
        map_val = metrics['map'].item() if isinstance(metrics['map'], torch.Tensor) else metrics['map']
        map50_val = metrics['map50'].item() if isinstance(metrics['map50'], torch.Tensor) else metrics['map50']
        
        # Calculate average F1 score across all classes
        f1_values = metrics['f1']
        if isinstance(f1_values, torch.Tensor):
            f1_values = f1_values.tolist()
        avg_f1 = sum(f1_values) / len(f1_values) if len(f1_values) > 0 else 0
        
        # Record results
        results_data['iou'].append(float(iou))
        results_data['map'].append(float(map_val))
        results_data['map50'].append(float(map50_val))
        results_data['f1'].append(float(avg_f1))
        
        # Determine which metric to use for optimization
        if metric == 'f1':
            metric_value = avg_f1
        elif metric == 'map50':
            metric_value = map50_val
        elif metric == 'map':
            metric_value = map_val
        else:
            # Default to F1
            metric_value = avg_f1
        
        # Update best threshold if selected metric is better
        if metric_value > best_score:
            best_score = metric_value
            best_iou = iou
    
    # Create DataFrame for easier analysis and visualization
    results_df = pd.DataFrame(results_data)
    
    return best_iou, results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to Faster R-CNN model checkpoint (.pth file)",
                        required=True)
    parser.add_argument("-cfg",
                        help="Path to data config file",
                        required=True)
    parser.add_argument("--output_dir",
                        help="Directory to save results",
                        default='threshold_optimization_results_rcnn')
    parser.add_argument("--test_imgs",
                        help="Path to test images directory",
                        required=True)
    parser.add_argument("--test_labels",
                        help="Path to test labels directory",
                        required=True)
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
    parser.add_argument("--stochastic",
                        action="store_true",
                        help="Use stochastic architecture for R-CNN",
                        default=False)
    parser.add_argument("--dropout_rate",
                        type=float,
                        default=0.5,
                        help="Dropout rate for stochastic models")
    parser.add_argument("--metric",
                        type=str,
                        choices=['f1', 'map', 'map50'],
                        default='f1',
                        help="Metric to optimize for: f1, map (mAP 0.5-0.95), or map50 (mAP at IoU=0.5)")

    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

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

    # Validate test directories
    if not os.path.exists(args.test_imgs):
        raise ValueError(f"Test images directory {args.test_imgs} not found")
    if not os.path.exists(args.test_labels):
        raise ValueError(f"Test labels directory {args.test_labels} not found")

    # First, find optimal confidence thresholds using default IoU
    default_iou = args.default_iou
    print(f"Finding optimal confidence thresholds using default IoU={default_iou} based on {args.metric} metric")
    best_confs, conf_results = find_optimal_conf_thresholds(
        args.pt, default_iou, class_names, args.test_imgs, args.test_labels, 
        device, args.output_dir, args.metric, conf_range
    )
    
    # Now find optimal IoU threshold using the optimized confidence thresholds
    print(f"Finding optimal IoU threshold using optimized confidence thresholds based on {args.metric} metric")
    best_iou, iou_results = find_optimal_iou_threshold(
        args.pt, best_confs, class_names, args.test_imgs, args.test_labels, 
        device, args.output_dir, args.metric, iou_range
    )
    
    _ = model_utils.save_optimization_results(
        best_iou, best_confs, class_names, args.metric, 
        conf_results, iou_results, args.output_dir
    )
    
    # Print optimal thresholds
    print(f"Optimal IoU threshold: {best_iou}")
    print("Optimal confidence thresholds:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {best_confs[i]}")
    
    # Final evaluation with optimal thresholds
    print("Running final evaluation with optimal thresholds")
    final_metrics = evaluate_with_thresholds(args.pt, best_iou, best_confs, class_names, args.test_imgs, args.test_labels, device, args.output_dir)

    f1_values = final_metrics['f1']
    if isinstance(f1_values, torch.Tensor):
        f1_values = f1_values.tolist()
    avg_f1 = sum(f1_values) / len(f1_values) if len(f1_values) > 0 else 0
    print(f"Final average F1 score: {avg_f1:.4f}")
    
    # Print the optimized metric value
    if args.metric == 'f1':
        print(f"Final optimized metric ({args.metric}): {avg_f1:.4f}")
    elif args.metric == 'map':
        print(f"Final optimized metric ({args.metric}): {final_metrics['map']:.4f}")
    elif args.metric == 'map50':
        print(f"Final optimized metric ({args.metric}): {final_metrics['map50']:.4f}")

if __name__ == "__main__":
    main() 