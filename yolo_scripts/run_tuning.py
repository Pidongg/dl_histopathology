from ultralytics import YOLO
import yaml
import argparse
from ray import tune



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

    search_space = {
        "degrees": tune.uniform(0.0,45.0),
        "shear": tune.uniform(0.0,10.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.9),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
    }
    # tune hyperparameters
    model.tune(space=search_space, use_ray=True, **cfg_args)


if __name__ == "__main__":
    main()
