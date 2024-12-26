from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import yaml
import argparse
import numpy as np


class AggressiveTuner(Tuner):
    def __init__(self, args=None):
        super().__init__(args={})
        self.mutation = 0.9
        self.sigma = 0.5
        
    def mutate(self, parent):
        """More aggressive mutation function"""
        r = np.random.random
        ng = len(parent)  # number of genes
        
        # Force larger mutations with higher probability
        v = np.zeros(ng)
        for i in range(ng):
            if r() < self.mutation:  # 90% chance of mutation
                # Generate larger changes
                change = r() * self.sigma * 5.0  # Increased multiplier
                if r() < 0.5:  # 50% chance of positive/negative
                    v[i] = 1.0 + change
                else:
                    v[i] = 1.0 - change
            else:
                v[i] = 1.0
                
        return parent * v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pretrained",
                        help="Path to pretrained model",
                        default='yolo11n.pt')
    parser.add_argument("-cfg", "--config",
                        help="Path to conf file for tuning",
                        required=True)

    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.pretrained)
    model.to('cuda')
    print("Device:", model.device.type)

    # Load config
    with open(args.config, "r") as stream:
        cfg_args = yaml.safe_load(stream)

    # Create aggressive tuner
    tuner = AggressiveTuner()
    
    # Add tuner to config
    cfg_args['tuner'] = tuner
    
    # Run tuning
    results = model.tune(**cfg_args)
    print(f"Best hyperparameters found: {results}")


if __name__ == "__main__":
    main()