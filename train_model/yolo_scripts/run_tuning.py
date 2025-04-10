from ultralytics import YOLO
import yaml
import argparse
import logging
import torch
import os
import ray
from ray import tune

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_gpu():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-pt", "--pretrained",
                           help="Path to pretrained model",
                           default='yolo11n.pt')
        parser.add_argument("-cfg", "--config",
                           help="Path to conf file for tuning",
                           required=True)
        args = parser.parse_args()

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Please check your CUDA setup.")
            return

        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        logger.info(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

        # Load base config
        with open(args.config, "r") as stream:
            try:
                cfg_args = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(f"Error loading config: {exc}")
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


        model = YOLO(cfg_args['pretrained'])
        model.tune(
            data=cfg_args['data'],
            space=search_space,
            epochs=cfg_args.get('epochs', 10),
            iterations=cfg_args.get('iterations', 100),
            use_ray=True,
            gpu_per_trial=4,
            **{k: v for k, v in cfg_args.items() if k not in ['data', 'epochs', 'iterations', 'pretrained'] and not k.startswith('space/')}
        )
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise
    finally:
        cleanup_gpu()
        ray.shutdown()  # Ensure Ray resources are released

if __name__ == "__main__":
    main()