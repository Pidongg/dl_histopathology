import torch
import ray
import os
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


   # Set CUDA devices to use all available GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Initialize Ray with GPU resources
    ray.init(num_gpus=4)  # Adjust based on available GPUs

    # Check CUDA availability
    if not torch.cuda.is_available():
        return  # get hyperparameters
    with open(cfg, "r") as stream:
        try:
            cfg_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    search_space = {
        "degrees": tune.uniform(1.5,45.0),
        "shear": tune.uniform(0.0,3.0),
        "translate": tune.uniform(0.0, 0.3),
        "flipud": tune.uniform(0.0, 0.3)
    }
    try:
        # Run tuning with valid YOLO arguments
        result_grid = model.tune(
            data=cfg_args['data'],
            space=search_space,
            use_ray=True,
            epochs=cfg_args.get('epochs', 10),
            iterations=cfg_args.get('iterations', 100),
            **{k: v for k, v in cfg_args.items() if k not in ['data', 'epochs', 'iterations']}
        )

    except Exception as e:
        raise


if __name__ == "__main__":
    main()
