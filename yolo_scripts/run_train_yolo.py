from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to pretrained model",
                        default='yolo11n.pt')
    parser.add_argument("-cfg",
                        help="Path to training configuration file",
                        default='/local/scratch/pz286/dl_histopathology/config/tau_training_config.yaml')

    args = parser.parse_args()

    pretrained_model = args.pt
    cfg = args.cfg

    # Load a pretrained YOLO model
    model = YOLO(pretrained_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # get hyperparameters
    with open(cfg, "r") as stream:
        try:
            cfg_args = yaml.safe_load(stream)
            # Print the crop_fraction from config
            LOGGER.info(f"Config crop_fraction: {cfg_args.get('crop_fraction', 'Not specified')}")
        except yaml.YAMLError as exc:
            print(exc)

    # Add debug callback to check image sizes
    def log_batch_stats(trainer):
        try:
            # Log at the start of training
            imgs = trainer.batch[0]
            LOGGER.info(f"Batch image shape: {imgs.shape}")
            LOGGER.info(f"Model args: {trainer.args}")
            if hasattr(trainer.args, 'crop_fraction'):
                LOGGER.info(f"Actual crop_fraction: {trainer.args.crop_fraction}")
            
            # Try to access other relevant attributes
            LOGGER.info(f"Batch size: {trainer.batch_size}")
            LOGGER.info(f"Device: {trainer.device}")
            
        except Exception as e:
            LOGGER.info(f"Debug info error: {str(e)}")
            
    # Add the callback to model
    model.add_callback("on_train_start", log_batch_stats)  # Changed to on_train_start

    # train models
    model.train(**cfg_args)

if __name__ == "__main__":
    main()