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
        
        # Log model parameters
        LOGGER.info(f"Image size: {trainer.args.imgsz}")
        LOGGER.info(f"Device: {trainer.device}")
        LOGGER.info(f"Number of classes: {trainer.model.nc}")
        
        # Log initial class distribution if available
        if hasattr(trainer.validator, 'dataloader'):
            val_loader = trainer.validator.dataloader
            all_targets = []
            for batch in val_loader:
                labels = batch['bboxes'].to(trainer.device)
                all_targets.append(labels)
            targets = torch.cat(all_targets, dim=0)
            class_counts = torch.bincount(targets[:, -1].long(), minlength=trainer.model.nc)
            LOGGER.info(f"Initial class distribution: {class_counts}")
            
    except Exception as e:
        LOGGER.error(f"Debug info error: {str(e)}")

def update_class_weights(trainer):
    """Callback to update class weights based on validation set distribution"""
    try:
        # Get validation dataloader
        val_loader = trainer.validator.dataloader
        
        # Collect all targets from validation set
        all_targets = []
        for batch in val_loader:
            labels = batch['bboxes'].to(trainer.device)
            all_targets.append(labels)
            
        # Concatenate all targets
        targets = torch.cat(all_targets, dim=0)
        
        # Get class labels
        y = targets[:, -1].cpu().numpy()  # Convert to numpy array
        classes = np.unique(y)  # Get unique classes
        
        # Calculate balanced weights using sklearn
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        
        # Convert to tensor and move to device
        class_weights = torch.from_numpy(class_weights).float().to(trainer.device)
        
        # Create new CrossEntropyLoss with updated weights
        trainer.model.loss.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none"
        )
        
        LOGGER.info(f"Updated class weights: {class_weights}")
        
    except Exception as e:
        LOGGER.error(f"Error updating class weights: {str(e)}")

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
    model.add_callback("on_val_end", update_class_weights)  # Update weights after each validation

    # Train model
    model.train(**cfg_args)

if __name__ == "__main__":
    main()