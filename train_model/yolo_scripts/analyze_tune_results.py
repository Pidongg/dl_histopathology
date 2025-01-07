from ultralytics import YOLO
import yaml
import argparse
import logging
import torch
import os
import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_function(config):
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

    # Extract hyperparameters
    hyp = {k: v for k, v in config.items() if k in [
        'degrees', 'shear', 'translate', 'scale', 'flipud', 'fliplr'
    ]}

    # Log the hyperparameters being used
    logger.info(f"Hyperparameters: {hyp}")

    # Call tune on the original model with cleaned hyperparameters
    try:
        results = model.module.tune(
            data=config['data'],
            space=hyp,
            epochs=config.get('epochs', 10),
            iterations=config.get('iterations', 100),
            plots=config.get('plots', True),
            save=True,  # Enable saving
            val=config.get('val', True),  # Enable validation
            **{k: v for k, v in config.items() if k not in ['data', 'epochs', 'iterations', 'pretrained'] 
               and not k in hyp}
        )
        
        # Report metrics for Ray Tune
        tune.report(
            fitness=results.get('fitness', 0),
            metrics_precision=results.get('metrics/precision(B)', 0),
            metrics_recall=results.get('metrics/recall(B)', 0),
            metrics_map50=results.get('metrics/mAP50(B)', 0),
            metrics_map50_95=results.get('metrics/mAP50-95(B)', 0),
            val_box_loss=results.get('val/box_loss', 0),
            val_cls_loss=results.get('val/cls_loss', 0),
            val_dfl_loss=results.get('val/dfl_loss', 0)
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

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Initialize Ray
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
        "degrees": tune.uniform(0.0, 45.0),
        "shear": tune.uniform(0.0, 10.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.9),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
    }

    # Add necessary parameters to config
    cfg_args['pretrained'] = args.pretrained

    try:
        # Run tuning with valid YOLO arguments
        tuner = tune.Tuner(
            train_function,
            param_space={**cfg_args, **search_space},
            run_config=RunConfig(
                name="yolo_tune",
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=1,  # Save checkpoint every iteration
                    checkpoint_at_end=True,  # Save checkpoint at the end
                    num_to_keep=None,  # Keep all checkpoints
                ),
            ),
            tune_config=tune.TuneConfig(
                metric="fitness",
                mode="max",
                num_samples=1  # Number of trials
            )
        )
        
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result(metric="fitness", mode="max")
        logger.info(f"Best trial config: {best_result.config}")
        logger.info(f"Best trial final fitness: {best_result.metrics['fitness']}")
        
        logger.info("Tuning completed successfully")
        
    except Exception as e:
        logger.error(f"Error during tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()