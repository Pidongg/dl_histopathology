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

def train_function(config):
    # Extract the hyperparameters from the config, removing the 'space/' prefix
    hyp = {k.replace('space/', ''): v for k, v in config.items() if k.startswith('space/')}
    
    # Check CUDA availability in each Ray worker
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Ray worker.")

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load model and perform training
    model = YOLO(config['pretrained'])
    model.to('cuda')
    model = torch.nn.DataParallel(model)

    # Log the cleaned hyperparameters
    logger.info(f"Hyperparameters after cleaning: {hyp}")

    # Call tune on the original model with cleaned hyperparameters
    try:
        model.module.tune(
            data=config['data'],
            space=hyp,  # Use the cleaned hyperparameters
            epochs=config.get('epochs', 10),
            iterations=config.get('iterations', 100),
            **{k: v for k, v in config.items() if k not in ['data', 'epochs', 'iterations', 'pretrained'] and not k.startswith('space/')}
        )
    except Exception as e:
        logger.error(f"Error during model tuning: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pretrained",
                       help="Path to pretrained model",
                       default='yolo11n.pt')
    parser.add_argument("-cfg", "--config",
                       help="Path to conf file for tuning",
                       required=True)
    args = parser.parse_args()

    # Set CUDA devices based on actual indices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Initialize Ray with GPU resources
    ray.init(num_gpus=4)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Please check your CUDA setup.")
        return

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
        "degrees": tune.uniform(1.5,45.0),
        "shear": tune.uniform(0.0,3.0),
        "translate": tune.uniform(0.0, 0.3),
        "flipud": tune.uniform(0.0, 0.3)
    }

    # Add necessary parameters to config
    cfg_args['pretrained'] = args.pretrained

    try:
        # Run tuning with valid YOLO arguments
        tune.run(
            train_function,
            config={**cfg_args, **search_space},  # Pass search space directly in config
            resources_per_trial={"cpu": 1, "gpu": 4},  # Allocate all 4 GPUs to each trial
        )
        
        logger.info("Tuning completed successfully")
        
    except Exception as e:
        logger.error(f"Error during tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()