import multiprocessing
# Must be at the very beginning of the script
multiprocessing.set_start_method('spawn', force=True)
import os
import subprocess
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import tqdm
import glob
import argparse
import yaml
import sys
import cv2
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preparation.rcnn_datasets import RCNNDataset

def get_gpu_info():
    """Get GPU utilization and memory info using nvidia-smi."""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8').strip()
        
        gpus = []
        for line in output.split('\n'):
            index, util, mem_used, mem_total = map(float, line.split(', '))
            gpus.append({
                'index': int(index),
                'utilization': util,
                'memory_used': mem_used,
                'memory_total': mem_total,
                'memory_free': mem_total - mem_used
            })
        return gpus
    except:
        return []

def select_best_gpus(num_gpus=1):
    """Select the least utilized GPUs based on both GPU utilization and memory usage."""
    gpus = get_gpu_info()
    if not gpus:
        return "0"  # Default to first GPU if can't get info
        
    # Sort GPUs by utilization (primary) and used memory (secondary)
    gpus.sort(key=lambda x: (x['utilization'], x['memory_used']))
    
    # Select the top N least utilized GPUs
    # selected_gpus = gpus[:num_gpus]
    selected_gpus = [3]
    return ','.join(str(gpu['index']) for gpu in selected_gpus)

def collate_fn(batch):
    return tuple(zip(*batch))


def get_unused_filename(out_dir, filename, extension):
    """
    Given an output directory and desired filename and extension, return a path in the directory
    that uses the filename + an index not already in use.

    Extension should be given with a '.', e.g. extension=".png".
    """
    matching_paths = glob.glob(f"{out_dir}/{filename}*{extension}")
    matching_paths.sort()
    if len(matching_paths) == 0:
        path_to_use = f"{out_dir}/{filename}{extension}"
    else:
        last_used_path = matching_paths[-1]
        last_used_filename = os.path.splitext(
            os.path.basename(last_used_path))[0]
        if last_used_filename == filename:
            path_to_use = f"{out_dir}/{filename}_0{extension}"
        else:
            idx = int(last_used_filename.split('_')[-1]) + 1
            path_to_use = f"{out_dir}/{filename}_{idx}{extension}"

    return path_to_use

def train_tune(config):
    def get_train_transform(config):
        return A.Compose([
            A.HorizontalFlip(p=config["horizontal_flip_p"]),
            A.VerticalFlip(p=config["vertical_flip_p"]),
            A.RandomRotate90(p=config["rotate90_p"]),
            A.Affine(
                rotate=(-config["rotate_degree"], config["rotate_degree"]),
                translate_percent=(-config["translate"], config["translate"]),
                scale=(1-config["scale"], 1+config["scale"]),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=config["affine_p"]
            ),
            A.CLAHE(p=config["clahe_p"], 
                    clip_limit=(config["clahe_clip_min"], config["clahe_clip_max"]),
                    tile_grid_size=(8, 8)),
            ToTensorV2(p=1.0),
        ], bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })

    def get_valid_transform():
        return A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })

    def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT")
            
        # Add dropout layers before the last 2 conv layers in the box head
        box_head = model.roi_heads.box_head
        layers = list(box_head.children())
        
        # Replace the box head with our modified version
        model.roi_heads.box_head = torch.nn.Sequential(*layers)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def train_one_epoch(model, optimizer, data_loader, device):
        """
        Written with reference to https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
        Function for running training iterations
        """
        model.train()
        epoch_loss = 0
        print('Training')

        # initialize tqdm progress bar
        prog_bar = tqdm.tqdm(data_loader, total=len(data_loader))

        for i, (images, targets) in enumerate(prog_bar):
            optimizer.zero_grad()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value

            losses.backward()
            optimizer.step()

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        return epoch_loss / len(data_loader)

    def validate(model, data_loader, device):
        """
        Function for evaluating performance on the validation set at each epoch
        """
        epoch_loss = 0
        print('Validating')

        # initialize tqdm progress bar
        prog_bar = tqdm.tqdm(data_loader, total=len(data_loader))

        with torch.no_grad():
            for i, (images, targets) in enumerate(prog_bar):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                epoch_loss += loss_value

                # update the loss value beside the progress bar for each iteration
                prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        return epoch_loss / len(data_loader)

    # Load config file inside the function
    cfg_path = config["cfg_path"]
    with open(cfg_path, "r") as stream:
        cfg_dict = yaml.safe_load(stream)
    
    # Set up paths
    path = cfg_dict['path']
    img_train_dir = os.path.join(path, cfg_dict['train'])
    img_val_dir = os.path.join(path, cfg_dict['val'])
    label_train_dir = os.path.join(path, cfg_dict['train_labels'])
    label_val_dir = os.path.join(path, cfg_dict['val_labels'])
    num_classes = len(cfg_dict['names'].keys()) + 1

    # Create device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create datasets with the tunable transforms
    dataset = RCNNDataset(
        img_dir=img_train_dir,
        label_dir=label_train_dir,
        width=512,
        height=512,
        transforms=get_train_transform(config),
        device=device
    )

    dataset_valid = RCNNDataset(
        img_dir=img_val_dir,
        label_dir=label_val_dir,
        width=512,
        height=512,
        transforms=get_valid_transform(),
        device=device
    )

    # Use the locally defined get_model_instance_segmentation
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Define data loaders with tunable batch size and proper CUDA settings
    data_loader_train = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Changed from 4 to 0 to avoid CUDA initialization issues
        collate_fn=collate_fn
    )

    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 to avoid CUDA initialization issues
        collate_fn=collate_fn
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=config["weight_decay"]
    )

    if config["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=2,
            gamma=0.2
        )
    else:  # cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["T_max"]
        )

    for epoch in range(6):  # Run for fewer epochs during tuning
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device)
        scheduler.step()
        valid_loss = validate(model, data_loader_valid, device)
        
        # Update to use the new reporting API
        from ray import train
        train.report({
            "valid_loss": valid_loss,
            "train_loss": train_loss
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-cfg",
                        help="Path to training configuration file",
                        required=True, default=None)

    parser.add_argument("model_save_dir",
                        help="Path to directory to save models to")

    parser.add_argument("model_name",
                        help="Base name under which to save the model")

    parser.add_argument("num_epochs", type=int,
                        help="Number of epochs to train for")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic dropout", default=False)

    args = parser.parse_args()

    cfg = args.cfg
    OUT_MODEL_DIR = args.model_save_dir
    model_name = args.model_name
    num_epochs = args.num_epochs

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("using device: ", device)

    if not os.path.exists(OUT_MODEL_DIR):
        os.makedirs(OUT_MODEL_DIR)

    # Convert config path to absolute path
    cfg = os.path.abspath(cfg)
    
    # Define hyperparameter search space
    config = {
        "cfg_path": cfg,  # Now using absolute path
        
        # Data augmentation params
        "horizontal_flip_p": tune.uniform(0.3, 0.7),
        "vertical_flip_p": tune.uniform(0.2, 0.7),
        "rotate90_p": tune.uniform(0.3, 0.7),
        "rotate_degree": tune.uniform(1.0, 50.0),
        "translate": tune.uniform(0.1, 0.3),
        "scale": tune.uniform(0, 0.3),
        "affine_p": tune.uniform(0.6, 0.9),
        "clahe_p": tune.uniform(0.8, 1.0),
        "clahe_clip_min": tune.uniform(1.0, 2.0),
        "clahe_clip_max": tune.uniform(3.0, 4.0),
        
        # Training params
        "weight_decay": tune.uniform(0.003, 0.005),
        "scheduler": tune.choice(["step", "cosine"]),
        "T_max": tune.choice([5, 8, 10]),
    }

    # Initialize Ray
    ray.init()

    # Define ASHA scheduler for early stopping with metric and mode
    scheduler = ASHAScheduler(
        max_t=100,  # Maximum number of training iterations
        grace_period=100,
        reduction_factor=2,
        metric="valid_loss",  # The metric to optimize
        mode="min"           # We want to minimize the validation loss
    )

    # Run hyperparameter tuning without duplicate metric and mode
    analysis = tune.run(
        train_tune,
        config=config,
        num_samples=100,
        scheduler=scheduler,
        resources_per_trial={"cpu": 0, "gpu": 1}
    )

    # Get best config and train final model
    best_config = analysis.get_best_config(metric="valid_loss", mode="min")
    print("Best hyperparameter configuration:", best_config)

    # Save best config
    with open(os.path.join(OUT_MODEL_DIR, "best_config.yaml"), "w") as f:
        yaml.dump(best_config, f)
