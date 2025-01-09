from ultralytics import YOLO
import yaml
import argparse
import torch

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

    # Train model
    model.train(**cfg_args)

if __name__ == "__main__":
    main()