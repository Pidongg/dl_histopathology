from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER

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

    # Add callback
    model.add_callback("on_batch_end", weight_monitor)

    # Train model
    model.train(**cfg_args)

if __name__ == "__main__":
    main()