# referenced the docs: https://docs.ultralytics.com/usage/python/

from ultralytics import YOLO
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to pretrained model",
                        default='yolov8n.pt')

    parser.add_argument("-cfg",
                        help="Path to training configuration file",
                        required=True, default=None)

    args = parser.parse_args()

    pretrained_model = args.pt
    cfg = args.cfg

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(pretrained_model)
    model.to('cuda')

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
