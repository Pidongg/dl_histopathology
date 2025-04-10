import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import yaml
from tqdm import tqdm
import json
from evaluator import RCNNEvaluator
import train_model.run_train_rcnn as run_train_rcnn

def evaluate_with_thresholds(checkpoint_path, data_yaml, iou_thresh, conf_thresholds, class_names, test_images, test_labels, device, save_dir):
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
    ap = evaluator.ap_per_class()
    map50 = evaluator.map50()
    map50_95 = evaluator.map50_95()
    
    return {
        'map': map50_95,
        'map50': map50,
        'maps': ap.tolist() if hasattr(ap, 'tolist') else ap,  # Per-class mAP
    }

def find_optimal_conf_thresholds(checkpoint_path, data_yaml, iou_thresh, class_names, test_images, test_labels, device, save_dir, conf_range=None):
    """Find optimal confidence threshold for each class."""
    if conf_range is None:
        conf_range = np.arange(0.05, 0.96, 0.1)  # Default if not specified
    
    best_confs = [0.25] * len(class_names)  # Default starting point
    best_map50s = [0] * len(class_names)  # Track best mAP50 for each class
    
    # Store dataframes for each class
    all_results_dfs = []
    
    # For each class, find optimal confidence threshold
    for class_idx in range(len(class_names)):
        print(f"Finding optimal threshold for class {class_idx}: {class_names[class_idx]}")
        
        # Initialize results data for this class
        results_data = {
            'class': [],
            'confidence': [],
            'map50': []
        }
        
        for conf in tqdm(conf_range):
            # Create conf thresholds with current conf for target class
            class_conf_thresholds = [0.25] * len(class_names)  # Default for other classes
            class_conf_thresholds[class_idx] = conf
            
            # Evaluate with current thresholds
            metrics = evaluate_with_thresholds(checkpoint_path, data_yaml, iou_thresh, class_conf_thresholds, class_names, test_images, test_labels, device, save_dir)
            
            # Extract per-class mAP50 values if available
            try:
                per_class_map50 = metrics['maps'][class_idx]
                # If it's a list (multiple IoU thresholds), take the value for IoU=0.5
                if isinstance(per_class_map50, list):
                    per_class_map50 = per_class_map50[0]  # First element corresponds to IoU=0.5
            except (AttributeError, IndexError, TypeError):
                # Fallback to using map50 (overall) for optimization
                per_class_map50 = metrics['map50']
            
            # Convert tensor to Python float if necessary
            if isinstance(per_class_map50, torch.Tensor):
                per_class_map50 = per_class_map50.item()
            
            # Record results
            results_data['class'].append(class_names[class_idx])
            results_data['confidence'].append(float(conf))
            results_data['map50'].append(float(per_class_map50))
            print(per_class_map50, best_map50s[class_idx])
            # Update best threshold if this class's mAP50 is better
            if per_class_map50 > best_map50s[class_idx]:
                best_map50s[class_idx] = per_class_map50
                best_confs[class_idx] = conf
        
        # Create DataFrame for this class and add to collection
        class_df = pd.DataFrame(results_data)
        all_results_dfs.append(class_df)
    
    # Combine all class results
    results_df = pd.concat(all_results_dfs, ignore_index=True)
    
    return best_confs, results_df

def find_optimal_iou_threshold(checkpoint_path, data_yaml, conf_thresholds, class_names, test_images, test_labels, device, save_dir, iou_range=None):
    """Find optimal IoU threshold using optimized confidence thresholds."""
    if iou_range is None:
        iou_range = np.arange(0.3, 0.71, 0.05)  # Default if not specified
    
    best_iou = 0.5  # Default
    best_map50 = 0
    
    results_data = {
        'iou': [],
        'map': [],
        'map50': [],
    }
    
    print("Finding optimal IoU threshold with optimized confidence thresholds")
    
    for iou in tqdm(iou_range):
        # Evaluate with current IoU threshold and optimized conf thresholds
        metrics = evaluate_with_thresholds(checkpoint_path, data_yaml, iou, conf_thresholds, class_names, test_images, test_labels, device, save_dir)
        
        # Convert tensor values to Python floats if necessary
        map_val = metrics['map'].item() if isinstance(metrics['map'], torch.Tensor) else metrics['map']
        map50_val = metrics['map50'].item() if isinstance(metrics['map50'], torch.Tensor) else metrics['map50']
        
        # Record results
        results_data['iou'].append(float(iou))
        results_data['map'].append(float(map_val))
        results_data['map50'].append(float(map50_val))
        
        # Update best threshold if mAP50 is better
        if map50_val > best_map50:
            best_map50 = map50_val
            best_iou = iou
    
    # Create DataFrame for easier analysis and visualization
    results_df = pd.DataFrame(results_data)
    
    return best_iou, results_df

def visualize_conf_thresholds(results_df, save_path='conf_threshold_optimization.png'):
    """Visualize confidence threshold optimization results."""
    plt.figure(figsize=(8, 6))
    
    # Create a color palette
    palette = sns.color_palette("husl", n_colors=len(results_df['class'].unique()))
    
    for i, class_name in enumerate(sorted(results_df['class'].unique())):
        class_data = results_df[results_df['class'] == class_name]
        
        plt.plot(class_data['confidence'], class_data['map50'], marker='o', label=class_name, color=palette[i])
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('mAP50')
    plt.title('mAP50 vs Confidence Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_iou_thresholds(results_df, save_path='iou_threshold_optimization.png'):
    """Visualize IoU threshold optimization results."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['iou'], results_df['map'], marker='o', linewidth=2)
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP (0.5-0.95)')
    plt.title('mAP vs IoU Threshold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['iou'], results_df['map50'], marker='o', linewidth=2)
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP50')
    plt.title('mAP50 vs IoU Threshold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

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

    # Load model
    
    # Get list of test images and labels
    if not os.path.exists(args.test_imgs):
        raise ValueError(f"Test images directory {args.test_imgs} not found")
    if not os.path.exists(args.test_labels):
        raise ValueError(f"Test labels directory {args.test_labels} not found")

    # First, find optimal confidence thresholds using default IoU
    default_iou = args.default_iou
    print(f"Finding optimal confidence thresholds using default IoU={default_iou} based on mAP50")
    best_confs, conf_results = find_optimal_conf_thresholds(
        args.pt, args.cfg, default_iou, class_names, args.test_imgs, args.test_labels, device, args.output_dir, conf_range
    )
    
    # Save confidence results
    conf_results.to_csv(os.path.join(args.output_dir, 'conf_optimization_results.csv'), index=False)
    
    # Visualize confidence results
    conf_vis_path = os.path.join(args.output_dir, 'conf_threshold_optimization.png')
    visualize_conf_thresholds(conf_results, conf_vis_path)
    
    # Now find optimal IoU threshold using the optimized confidence thresholds
    print(f"Finding optimal IoU threshold using optimized confidence thresholds based on mAP50")
    best_iou, iou_results = find_optimal_iou_threshold(args.pt, args.cfg, best_confs, class_names, args.test_imgs, args.test_labels, device, args.output_dir, iou_range
    )
    
    # Save IoU results
    iou_results.to_csv(os.path.join(args.output_dir, 'iou_optimization_results.csv'), index=False)
    
    # Visualize IoU results
    iou_vis_path = os.path.join(args.output_dir, 'iou_threshold_optimization.png')
    visualize_iou_thresholds(iou_results, iou_vis_path)
    
    # Save optimal thresholds
    optimal_thresholds = {
        'iou_threshold': float(best_iou),
        'confidence_thresholds': {class_names[i]: float(best_confs[i]) for i in range(len(class_names))}
    }
    
    with open(os.path.join(args.output_dir, 'optimal_thresholds.json'), 'w') as f:
        json.dump(optimal_thresholds, f, indent=4)
    
    # Print optimal thresholds
    print(f"Optimal IoU threshold: {best_iou}")
    print("Optimal confidence thresholds:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {best_confs[i]}")
    
    # Final evaluation with optimal thresholds
    print("Running final evaluation with optimal thresholds")
    final_metrics = evaluate_with_thresholds(args.pt, args.cfg, best_iou, best_confs, class_names, args.test_imgs, args.test_labels, device, args.output_dir)
    
    print(f"Final mAP50-95: {final_metrics['map']:.4f}")
    print(f"Final mAP50: {final_metrics['map50']:.4f}")

if __name__ == "__main__":
    main() 