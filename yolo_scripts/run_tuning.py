from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import yaml
import argparse


class AggressiveTuner(Tuner):
    def __init__(self, args=None):  # Make args optional
        super().__init__(args={})    # Initialize with empty dict
        self.mutation = 0.9    
        self.sigma = 0.5       


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pretrained",
                        help="Path to pretrained model",
                        default='yolo11n.pt')

    parser.add_argument("-cfg", "--config",
                        help="Path to conf file for tuning",
                        required=True, default=None)

    args = parser.parse_args()

    pretrained_model = args.pretrained
    cfg = args.config

    # Load a pretrained YOLO model
    model = YOLO(pretrained_model)
    model.to('cuda')

    print("Device:", model.device.type)

    # get hyperparameters
    with open(cfg, "r") as stream:
        try:
            cfg_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    # Create tuner with aggressive settings
    tuner = AggressiveTuner()
    tuner.mutation = 0.9
    tuner.sigma = 0.5

    # Run tuning
    results = model.tune(**cfg_args)  # Pass config directly to tune()

    print(f"Best hyperparameters found: {results}")


if __name__ == "__main__":
    main()