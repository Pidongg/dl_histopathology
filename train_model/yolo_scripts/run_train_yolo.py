from ultralytics import YOLO
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to pretrained model",
                        default='yolo11n.pt')
    parser.add_argument("-cfg",
                        help="Path to training configuration file",
                        default='/local/scratch/pz286/dl_histopathology/config/train_config_final.yaml')

    args = parser.parse_args()
    
    model = YOLO(args.pt)

    # Load config
    with open(args.cfg, "r") as stream:
        cfg_args = yaml.safe_load(stream)
    # Train model
    model.train(**cfg_args)

if __name__ == "__main__":
    main()