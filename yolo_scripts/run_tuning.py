from ultralytics import YOLO
import yaml
import argparse
import logging
from ray import tune

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pretrained",
                       help="Path to pretrained model",
                       default='yolo8n.pt')
    parser.add_argument("-cfg", "--config",
                       help="Path to conf file for tuning",
                       required=True)
    args = parser.parse_args()

    # Load base config
    with open(args.config, "r") as stream:
        try:
            cfg_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading config: {exc}")
            return

    # Define search space following documentation
    search_space = {
        # Augmentation parameters (simplified for testing)
        "degrees": tune.uniform(0.0, 45.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.9),
        "shear": tune.uniform(0.0, 10.0),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
    }

    # Load model
    model = YOLO(args.pretrained)
    
    try:
        # Run tuning with only valid YOLO arguments
        result_grid = model.tune(
            data=cfg_args['data'],
            space=search_space,
            use_ray=True,  # Enable Ray backend
            epochs=cfg_args.get('epochs', 10),
            iterations=cfg_args.get('iterations', 100),
            plots=cfg_args.get('plots', True),
            save=cfg_args.get('save', False),
            val=cfg_args.get('val', True)
        )
        
        logger.info("Tuning completed successfully")
        logger.info(f"Best config: {result_grid}")
        
    except Exception as e:
        logger.error(f"Error during tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()