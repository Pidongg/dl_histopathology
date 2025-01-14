from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pdq_evaluation')))
from read_files import convert_yolo_to_rvc

def plot_confusion_matrix(confusion_matrix, class_names):
    """Plot confusion matrix using seaborn"""
    # Get the matrix data and resize it to match the number of classes
    array = confusion_matrix.matrix
    n_classes = len(class_names)
    array = array[:n_classes, :n_classes]  # Take only the relevant classes
    
    df_cm = pd.DataFrame(array, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def save_predictions_to_json(model, data_yaml, conf_thresh, save_path):
    """Save model predictions to JSON in YOLO format"""
    results_dict = {}
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get base path and validation images path
    base_path = data_config.get('path', '')
    val_path = data_config.get('val', '')
    
    # Try both absolute and relative paths
    full_val_paths = [
        Path(base_path) / val_path,
        Path(val_path),
        Path(os.path.dirname(data_yaml)) / val_path
    ]
    
    valid_path = next((path for path in full_val_paths if path.exists()), None)
    if valid_path is None:
        LOGGER.error("No valid validation path found!")
        return
    
    # List all image files recursively
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(valid_path.rglob(ext)))
    
    # Run predictions on validation set
    for img_file in image_files:
        try:
            results = model.predict(str(img_file), conf=conf_thresh)[0]
            boxes = results.boxes
            
            predictions = []
            if len(boxes) > 0:
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        conf = float(box.conf.cpu().numpy()[0])
                        cls_id = int(box.cls.cpu().numpy()[0])
                        predictions.append([float(x1), float(y1), float(x2), float(y2), conf, cls_id])
                    except Exception as e:
                        LOGGER.warning(f"Error processing box: {e}")
                        continue
            
            rel_path = str(img_file.relative_to(valid_path))
            results_dict[rel_path] = predictions
            
        except Exception as e:
            LOGGER.warning(f"Error processing image {img_file}: {e}")
            continue
    
    # Save to JSON
    try:
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    except Exception as e:
        LOGGER.error(f"Error saving predictions to {save_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to pretrained model",
                        default='yolo11n.pt')
    parser.add_argument("-cfg",
                        help="Path to data config file",
                        default='/local/scratch/pz286/dl_histopathology/config/tau_data_test.yaml')
    parser.add_argument("-conf", 
                        help='confidence threshold',
                        default=0.25)
    parser.add_argument("-iou", 
                        help='iou threshold',
                        default=0.5)
    parser.add_argument("-save_json",
                        help='Path to save YOLO format predictions JSON',
                        default='predictions_yolo.json')
    parser.add_argument("-save_rvc",
                        help='Path to save RVC format predictions JSON',
                        default='predictions_rvc.json')
    parser.add_argument("--save_detections", 
                        action='store_true',
                        help='Whether to save detection results to JSON files')

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

    # Save predictions only if flag is set
    if args.save_detections:
        LOGGER.info("Saving predictions to JSON files...")
        save_predictions_to_json(model, args.cfg, float(args.conf), args.save_json)
        convert_yolo_to_rvc(args.save_json, args.save_rvc, class_names)
        LOGGER.info("Finished saving predictions")
    
    # Run validation and get metrics
    LOGGER.info("Running validation...")
    metrics = model.val(data=args.cfg, conf=float(args.conf), iou=float(args.iou))
    
    # Print metrics
    LOGGER.info(f"mAP50-95: {metrics.box.map:.4f}")
    LOGGER.info(f"mAP50: {metrics.box.map50:.4f}")
    LOGGER.info(f"mAP75: {metrics.box.map75:.4f}")
    LOGGER.info(f"Per-class mAP50-95: {metrics.box.maps}")

    # Plot confusion matrix
    confusion_matrix = metrics.confusion_matrix
    plot_confusion_matrix(confusion_matrix, class_names)

if __name__ == "__main__":
    main()