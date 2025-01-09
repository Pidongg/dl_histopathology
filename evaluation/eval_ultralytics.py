from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def weight_monitor(trainer):
    """Callback to monitor weight updates and dataset coverage"""
    if not hasattr(trainer, 'batch_counts'):
        trainer.batch_counts = {}
    
    # Count batches per epoch
    if trainer.epoch not in trainer.batch_counts:
        trainer.batch_counts[trainer.epoch] = 0
    trainer.batch_counts[trainer.epoch] += 1
    
    # Print at end of each epoch
    if trainer.batch == trainer.trainer.nb - 1:  # Last batch of epoch
        print(f"\nEpoch {trainer.epoch} saw {trainer.batch_counts[trainer.epoch]} batches")
        print(f"Total images seen: {trainer.batch_counts[trainer.epoch] * trainer.args.batch}")

def plot_confusion_matrix(confusion_matrix, class_names):
    """Plot confusion matrix using seaborn"""
    array = confusion_matrix.cpu().numpy()
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

    args = parser.parse_args()

    # Load model and config
    model = YOLO(args.pt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    metrics = model.val(data=args.cfg, conf=0.25, iou=0.5)
    
    # Print metrics
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    print(f"Per-class mAP50-95: {metrics.box.maps}")

    # Get and plot confusion matrix
    confusion_matrix = metrics.confusion_matrix
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    
    plot_confusion_matrix(confusion_matrix, class_names)

if __name__ == "__main__":
    main()