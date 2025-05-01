from ultralytics import YOLO
import yaml
import argparse
import torch
import os
import ray
from ray import tune
import random

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-pt", "--pretrained",
                           help="Path to pretrained model",
                           default='yolo11n.pt')
        parser.add_argument("-cfg", "--config",
                           help="Path to conf file for tuning",
                           required=True)
        parser.add_argument("-gpu", "--gpu_id",
                           help="Specific GPU ID to use (e.g., 0,1)",
                           default=None)
        args = parser.parse_args()

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA is not available. Please check your CUDA setup.")
            return

        # Set GPU visibility if specified
        if args.gpu_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
            print(f"Setting CUDA_VISIBLE_DEVICES to {args.gpu_id}")

        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        print(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

        # Load base config
        with open(args.config, "r") as stream:
            try:
                cfg_args = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"Error loading config: {exc}")
                return

        # Define search space
        search_space = {
            "degrees": tune.uniform(1.5,40.0),
            "shear": tune.uniform(0.0,3.0),
            "translate": tune.uniform(0.0, 0.3),
            "flipud": tune.uniform(0.0, 0.4),
            "fliplr": tune.uniform(0.4, 0.6),
            "auto_augment": tune.choice([True, False]),
            "scale": tune.uniform(0.1, 0.3),
        }

        # Add necessary parameters to config
        cfg_args['pretrained'] = args.pretrained

        # Generate a random port for Ray to avoid conflicts with existing Ray instances
        ray_port = random.randint(10000, 20000)
        print(f"Starting Ray on port {ray_port}")
        
        # Initialize Ray with a specific port
        if not ray.is_initialized():
            print("Initializing Ray")
            ray.init(ignore_reinit_error=True, 
                    _redis_password="password", 
                    dashboard_port=ray_port, 
                    include_dashboard=True)

        model = YOLO(cfg_args['pretrained'])
        model.tune(
            data=cfg_args['data'],
            space=search_space,
            epochs=cfg_args.get('epochs', 10),
            iterations=cfg_args.get('iterations', 100),
            use_ray=True,
            gpu_per_trial=1,
            **{k: v for k, v in cfg_args.items() if k not in ['data', 'epochs', 'iterations', 'pretrained'] and not k.startswith('space/')}
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()