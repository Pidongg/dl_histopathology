from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def log_batch_stats(trainer):
    """Callback to log training configuration and batch statistics at start of training"""
    try:
        # Log training configuration
        LOGGER.info(f"Training args: {trainer.args}")
        LOGGER.info(f"Data args: {trainer.data}")
        LOGGER.info(f"Image size: {trainer.args.imgsz}")
        LOGGER.info(f"Device: {trainer.device}")
        LOGGER.info(f"Number of classes: {trainer.model.nc}")
    except Exception as e:
        LOGGER.error(f"Debug info error: {str(e)}")

def update_class_weights_batch(trainer):
    """Callback to update class weights based on current batch"""
    try:
        # Get current batch targets
        targets = trainer.batch['bboxes'].to(trainer.device)
        
        if len(targets) == 0:
            return
            
        # Get class labels from current batch
        y = targets[:, -1].cpu().numpy()
        classes = np.arange(trainer.model.nc)  # Use all possible classes
        
        # Calculate balanced weights using sklearn
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        
        # Convert to tensor and move to device
        class_weights = torch.from_numpy(class_weights).float().to(trainer.device)
        
        # Update CrossEntropyLoss with new weights
        trainer.model.loss.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none"
        )
        
        if trainer.epoch == 0 and trainer.batch_idx % 100 == 0:  # Log less frequently
            LOGGER.info(f"Batch {trainer.batch_idx} class weights: {class_weights}")
        
    except Exception as e:
        LOGGER.error(f"Error updating batch weights: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to pretrained model",
                        default='yolo11n.pt')
    parser.add_argument("-cfg",
                        help="Path to training configuration file",
                        default='/local/scratch/pz286/dl_histopathology/config/tau_training_config.yaml')

    args = parser.parse_args()

    # Load model and config
    model = YOLO(args.pt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load config
    with open(args.cfg, "r") as stream:
        cfg_args = yaml.safe_load(stream)

    # Add callbacks
    model.add_callback("on_train_start", log_batch_stats)
    model.add_callback("on_train_batch_start", update_class_weights_batch)  # Update weights before each batch

    # Train model
    model.train(**cfg_args)

if __name__ == "__main__":
    main()