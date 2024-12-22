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
        if trainer.epoch == 0 and trainer.batch == 0:
            batch = trainer.batch_idx
            imgs = trainer.batch[0]
            LOGGER.info(f"First batch image shape: {imgs.shape}")
            LOGGER.info(f"Model args: {trainer.args}")
            
    # Add the callback to model
    model.add_callback("on_train_batch_start", log_batch_stats)

    # train models
    model.train(**cfg_args)

if __name__ == "__main__":
    main()