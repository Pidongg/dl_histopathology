from ultralytics import YOLO
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from ultralytics.utils import LOGGER
import json

def evaluate_with_thresholds(model, data_yaml, iou_thresh, conf_thresholds, class_names):
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

def find_optimal_conf_thresholds(model, data_yaml, iou_thresh, class_names, conf_range=None):
    """Find optimal confidence threshold for each class."""
    if conf_range is None:
        conf_range = np.arange(0.05, 0.96, 0.1)  # Default if not specified
    
    best_confs = [0.25] * len(class_names)  # Default starting point
    best_map50s = [0] * len(class_names)  # Track best mAP50 for each class
    
    # Store dataframes for each class
    all_results_dfs = []
    
    # For each class, find optimal confidence threshold
    for class_idx in range(len(class_names)):
        LOGGER.info(f"Finding optimal threshold for class {class_idx}: {class_names[class_idx]}")
        
        # Initialize results data for this class
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
            class_conf_thresholds = [0.25] * len(class_names)  # Default for other classes
            class_conf_thresholds[class_idx] = conf
            
            # Evaluate with current thresholds
            metrics = evaluate_with_thresholds(model, data_yaml, iou_thresh, class_conf_thresholds, class_names)
            
            # Try to extract per-class AP@0.5 values if available
            try:
                # If ap50 exists and contains per-class values
                per_class_map50 = metrics.box.ap50[class_idx]
            except (AttributeError, IndexError):
                # Fallback to using map50 (overall) for optimization
                per_class_map50 = metrics['map50']
            
            # Record results
            results_data['class'].append(class_names[class_idx])
            results_data['confidence'].append(conf)
            results_data['f1'].append(metrics['f1'][class_idx])
            results_data['precision'].append(metrics['precision'][class_idx])
            results_data['recall'].append(metrics['recall'][class_idx])
            results_data['map50'].append(per_class_map50)
            
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

def find_optimal_iou_threshold(model, data_yaml, conf_thresholds, class_names, iou_range=None):
    """Find optimal IoU threshold using optimized confidence thresholds."""
    if iou_range is None:
        iou_range = np.arange(0.3, 0.71, 0.05)  # Default if not specified
    
    best_iou = 0.5  # Default
    best_map50 = 0  # Changed from best_map to best_map50
    
    results_data = {
        'iou': [],
        'map': [],
        'map50': [],
        'map75': []
    }
    
    LOGGER.info("Finding optimal IoU threshold with optimized confidence thresholds")
    
    for iou in tqdm(iou_range):
        # Evaluate with current IoU threshold and optimized conf thresholds
        metrics = evaluate_with_thresholds(model, data_yaml, iou, conf_thresholds, class_names)
        
        # Record results
        results_data['iou'].append(iou)
        results_data['map'].append(metrics['map'])
        results_data['map50'].append(metrics['map50'])
        results_data['map75'].append(metrics['map75'])
        
        # Update best threshold if mAP50 is better (changed from mAP)
        if metrics['map50'] > best_map50:
            best_map50 = metrics['map50']
            best_iou = iou
    
    # Create DataFrame for easier analysis and visualization
    results_df = pd.DataFrame(results_data)
    
    return best_iou, results_df

def visualize_conf_thresholds(results_df, save_path='conf_threshold_optimization.png'):
    """Visualize confidence threshold optimization results."""
    plt.figure(figsize=(15, 10))
    
    # Create a color palette
    palette = sns.color_palette("husl", n_colors=len(results_df['class'].unique()))
    
    for i, class_name in enumerate(sorted(results_df['class'].unique())):
        class_data = results_df[results_df['class'] == class_name]
        
        plt.subplot(2, 2, 1)
        plt.plot(class_data['confidence'], class_data['f1'], marker='o', label=class_name, color=palette[i])
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(class_data['confidence'], class_data['precision'], marker='o', color=palette[i])
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(class_data['confidence'], class_data['recall'], marker='o', color=palette[i])
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(class_data['confidence'], class_data['map50'], marker='o', color=palette[i])
        plt.xlabel('Confidence Threshold')
        plt.ylabel('mAP50')
        plt.title('mAP50 vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
    
    # Add legend to the first subplot
    plt.subplot(2, 2, 1)
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
    plt.plot(results_df['iou'], results_df['map50'], marker='o', linewidth=2, label='mAP50')
    plt.plot(results_df['iou'], results_df['map75'], marker='o', linewidth=2, label='mAP75')
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP')
    plt.title('mAP50 and mAP75 vs IoU Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

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

    args = parser.parse_args()

    # Load model and config
    model = YOLO(args.pt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load class names
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    LOGGER.info(f"Loaded {len(class_names)} classes")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define threshold ranges using step sizes
    # Add a small epsilon to max value to ensure it's included in the range
    conf_range = np.arange(args.conf_min, args.conf_max + args.conf_step/2, args.conf_step)
    iou_range = np.arange(args.iou_min, args.iou_max + args.iou_step/2, args.iou_step)

    # First, find optimal confidence thresholds using default IoU
    default_iou = args.default_iou
    LOGGER.info(f"Finding optimal confidence thresholds using default IoU={default_iou} based on mAP50")
    best_confs, conf_results = find_optimal_conf_thresholds(
        model, args.cfg, default_iou, class_names, conf_range
    )
    
    # Save confidence results
    conf_results.to_csv(os.path.join(args.output_dir, 'conf_optimization_results.csv'), index=False)
    
    # Visualize confidence results
    conf_vis_path = os.path.join(args.output_dir, 'conf_threshold_optimization.png')
    visualize_conf_thresholds(conf_results, conf_vis_path)
    
    # Now find optimal IoU threshold using the optimized confidence thresholds
    LOGGER.info(f"Finding optimal IoU threshold using optimized confidence thresholds based on mAP50")
    best_iou, iou_results = find_optimal_iou_threshold(
        model, args.cfg, best_confs, class_names, iou_range
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
    LOGGER.info(f"Optimal IoU threshold: {best_iou}")
    LOGGER.info("Optimal confidence thresholds:")
    for i, class_name in enumerate(class_names):
        LOGGER.info(f"  {class_name}: {best_confs[i]}")
    
    # Final evaluation with optimal thresholds
    LOGGER.info("Running final evaluation with optimal thresholds")
    final_metrics = evaluate_with_thresholds(model, args.cfg, best_iou, best_confs, class_names)
    
    LOGGER.info(f"Final mAP50-95: {final_metrics['map']:.4f}")
    LOGGER.info(f"Final mAP50: {final_metrics['map50']:.4f}")
    LOGGER.info(f"Final per-class F1 scores: {final_metrics['f1']}")

if __name__ == "__main__":
    main()