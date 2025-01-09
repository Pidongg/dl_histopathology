from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

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