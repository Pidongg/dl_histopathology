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
                        default='C:/Users/peiya/Desktop/dissertation/dl_histopathology/config/tau_training_config.yaml')

    args = parser.parse_args()

    pretrained_model = args.pt
    cfg = args.cfg

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(pretrained_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print("Device:", model.device.type)

    # get hyperparameters
    with open(cfg, "r") as stream:
        try:
            cfg_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # train models
    model.train(**cfg_args)


if __name__ == "__main__":
    main()
