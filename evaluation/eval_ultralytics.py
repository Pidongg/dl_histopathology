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
    # Initialize results dictionary
    results_dict = {}
    
    # Get dataset from YAML
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get validation images
    val_images = data_config.get('val', '')
    if isinstance(val_images, str):
        val_images = [val_images]
    
    # Run predictions on validation set
    for img_path in val_images:
        img_dir = Path(img_path)
        if img_dir.is_dir():
            for img_file in img_dir.glob('*.[jp][pn][gf]'):  # Match jpg, png, jpeg
                results = model.predict(str(img_file), conf=conf_thresh)[0]
                boxes = results.boxes
                if len(boxes) > 0:
                    # Convert boxes to desired format [x1, y1, x2, y2, conf, class_id]
                    predictions = []
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        conf = float(box.conf.cpu().numpy()[0])
                        cls_id = int(box.cls.cpu().numpy()[0])
                        predictions.append([float(x1), float(y1), float(x2), float(y2), conf, cls_id])
                    results_dict[img_file.name] = predictions
                else:
                    results_dict[img_file.name] = []
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Saved predictions to {save_path}")

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

    args = parser.parse_args()

    # Load model and config
    model = YOLO(args.pt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load class names first to ensure they match the model
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    # Save predictions to YOLO format JSON
    save_predictions_to_json(model, args.cfg, float(args.conf), args.save_json)
    
    # Convert to RVC format
    convert_yolo_to_rvc(args.save_json, args.save_rvc, class_names)
    
    # Run validation and get metrics
    metrics = model.val(data=args.cfg, conf=float(args.conf), iou=float(args.iou))
    
    # Print metrics
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    print(f"Per-class mAP50-95: {metrics.box.maps}")

    # Get and plot confusion matrix
    confusion_matrix = metrics.confusion_matrix
    print(f"Confusion matrix shape: {confusion_matrix.matrix.shape}")
    
    plot_confusion_matrix(confusion_matrix, class_names)

if __name__ == "__main__":
    main()