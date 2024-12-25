from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import yaml
import argparse


class AggressiveTuner(Tuner):
    def __init__(self, args):
        super().__init__(args)
        self.mutation = 0.9    # Increased from 0.8
        self.sigma = 0.5       # Increased from 0.2
        
    def mutate(self, parent):
        # Override the mutation method to be more aggressive
        child = super().mutate(parent)
        return child


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

    # Set up aggressive tuner
    tuner = AggressiveTuner(cfg_args)
    
    # Run tuning with aggressive exploration
    model.tune(tuner=tuner, **cfg_args)


if __name__ == "__main__":
    main()