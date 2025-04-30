from data_preparation.rcnn_datasets import RCNNDataset
import threading
import time
import matplotlib.pyplot as plt
from ray.tune.schedulers import ASHAScheduler
from ray import tune, train
import ray
import cv2
import sys
import yaml
import argparse
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
import subprocess
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import common functions from run_train_rcnn.py
from train_model.rcnn_scripts.run_train_rcnn import collate_fn, get_valid_transform, train_one_epoch, validate

def renew_kerberos_ticket(password, interval=10800):
    """
    Periodically renew Kerberos ticket to prevent expiration.

    Args:
        password: The Kerberos password
        interval: Time in seconds between renewal attempts (default 30 minutes)
    """
    def renewal_worker():
        while True:
            try:
                # Run kinit with the password
                process = subprocess.Popen(
                    ['kinit'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.communicate(input=password.encode())
                print("Kerberos ticket renewed successfully")
            except Exception as e:
                print(f"Error renewing Kerberos ticket: {e}")

            # Sleep for the specified interval
            time.sleep(interval)

    # Start renewal in a separate thread
    renewal_thread = threading.Thread(target=renewal_worker, daemon=True)
    renewal_thread.start()
    return renewal_thread


def get_train_transform(config):
    """
    Defines the train transforms for the model with tunable parameters.
    """
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
        A.MotionBlur(
            blur_limit=config["motion_blur_limit"], p=config["motion_blur_p"]),
        A.MedianBlur(
            blur_limit=config["median_blur_limit"], p=config["median_blur_p"]),
        A.Blur(blur_limit=config["blur_limit"], p=config["blur_p"]),
        A.CLAHE(p=config["clahe_p"],
                clip_limit=(config["clahe_clip_min"],
                            config["clahe_clip_max"]),
                tile_grid_size=(8, 8)),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_model_instance_segmentation(num_classes):
    """
    Defines the instance segmentation model.
    Using a simplified version for tuning compared to the one in run_train_rcnn.py
    """
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
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)

    return model


def train_tune(config):
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
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=2,
        shuffle=False,
        num_workers=0,
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

    start_epoch = 0

    # Load checkpoint if it exists
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        checkpoint_path = os.path.join(loaded_checkpoint, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")

    # Use epochs_per_trial from the config instead of hardcoded value
    for epoch in range(start_epoch, config["epochs_per_trial"]):
        train_loss = train_one_epoch(
            model, optimizer, data_loader_train, device)
        scheduler.step()
        valid_loss = validate(model, data_loader_valid, device)

        # Save checkpoint
        checkpoint_dir = train.get_checkpoint()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss
        }

        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)

        # Update to use the new reporting API with checkpoint
        train.report({
            "valid_loss": valid_loss,
            "train_loss": train_loss
        }, checkpoint=checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-cfg",
                        help="Path to training configuration file",
                        required=True, default=None)

    parser.add_argument("model_save_dir",
                        help="Path to directory to save models to")

    parser.add_argument("model_name",
                        help="Base name under which to save the model")

    parser.add_argument("epochs_per_trial", type=int,
                        help="Number of epochs to train each trial")

    parser.add_argument("num_trials", type=int,
                        help="Number of hyperparameter trials to run")

    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic dropout", default=False)
    parser.add_argument("--kerberos-password",
                        help="Kerberos password for ticket renewal", default=None)
    parser.add_argument("--resume-path",
                        help="Directory name of the previous experiment to resume (e.g., 'train_tune_2025-04-13_22-33-48'). "
                             "Can be either the directory name within storage_path or a full path.",
                        default=None)
    parser.add_argument("--resume-checkpoint", action="store_true",
                        help="Use 'CHECKPOINT_DIR' mode instead of experiment directory. "
                             "This tells Ray to directly restore from a specific checkpoint.",
                        default=False)

    args = parser.parse_args()

    cfg = args.cfg
    OUT_MODEL_DIR = args.model_save_dir
    model_name = args.model_name
    epochs_per_trial = args.epochs_per_trial
    num_trials = args.num_trials
    kerberos_password = args.kerberos_password
    resume_path = args.resume_path

    # Start Kerberos ticket renewal in background
    renewal_thread = renew_kerberos_ticket(kerberos_password)
    print("Started Kerberos ticket renewal process")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("using device: ", device)

    if not os.path.exists(OUT_MODEL_DIR):
        os.makedirs(OUT_MODEL_DIR)

    # Convert config path to absolute path
    cfg = os.path.abspath(cfg)

    # Define hyperparameter search space
    config = {
        "cfg_path": cfg,  # Now using absolute path
        "epochs_per_trial": epochs_per_trial,  # Add epochs_per_trial to config

        # Data augmentation params
        "horizontal_flip_p": tune.uniform(0.0, 0.7),
        "vertical_flip_p": tune.uniform(0.0, 0.7),
        "rotate90_p": tune.uniform(0.0, 0.7),
        "rotate_degree": tune.uniform(1.0, 50.0),
        "translate": tune.uniform(0.0, 0.3),
        "scale": tune.uniform(0.0, 0.3),
        "affine_p": tune.uniform(0.0, 0.9),

        # Blur augmentation params
        "motion_blur_limit": tune.choice([3, 5, 7]),
        "motion_blur_p": tune.uniform(0.0, 0.5),
        "median_blur_limit": tune.choice([3, 5, 7]),
        "median_blur_p": tune.uniform(0.0, 0.4),
        "blur_limit": tune.choice([3, 5, 7]),
        "blur_p": tune.uniform(0.0, 0.4),

        # CLAHE params
        "clahe_p": tune.uniform(0.0, 1.0),
        "clahe_clip_min": tune.uniform(1.0, 2.0),
        "clahe_clip_max": tune.uniform(2.0, 4.0),

        # Training params
        "weight_decay": tune.uniform(0.0003, 0.0008),
        "scheduler": tune.choice(["step", "cosine"]),
        "T_max": tune.choice([5, 8, 10]),
    }

    ray.init()

    # Define ASHA scheduler for early stopping with metric and mode
    scheduler = ASHAScheduler(
        max_t=epochs_per_trial,  # Maximum number of training iterations
        # Allow at least 3 epochs before stopping
        grace_period=min(epochs_per_trial, 3),
        reduction_factor=2,
        metric="valid_loss",  # The metric to optimize
        mode="min"           # We want to minimize the validation loss
    )

    analysis = tune.run(
        train_tune,
        config=config,
        num_samples=num_trials,  # Use num_trials instead of num_epochs
        scheduler=scheduler,
        resources_per_trial={"cpu": 0, "gpu": 1},
        resume="AUTO+ERRORED",
        keep_checkpoints_num=2,  # Keep the 2 most recent checkpoints
        storage_path=os.path.abspath(
            os.path.join(OUT_MODEL_DIR, "ray_results")),
        name=resume_path
    )

    # Get best config and train final model
    best_config = analysis.get_best_config(metric="valid_loss", mode="min")
    print("Best hyperparameter configuration:", best_config)

    # Save best config
    with open(os.path.join(OUT_MODEL_DIR, "best_config.yaml"), "w") as f:
        yaml.dump(best_config, f)

    # Add visualization for hyperparameter tuning
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(OUT_MODEL_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Plot training curves for all trials
    dfs = analysis.trial_dataframes
    fig, ax = plt.subplots(figsize=(12, 8))
    for trial_id, df in dfs.items():
        if 'valid_loss' in df.columns:
            ax.plot(df['training_iteration'], df['valid_loss'],
                    alpha=0.3, label=trial_id[:6])

    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Across Trials')
    plt.savefig(os.path.join(plots_dir, "all_trials_validation_loss.png"))
    plt.close()

    # 2. Plot parameter importance
    param_importance = analysis.get_all_configs()
    param_scores = []

    for param in best_config.keys():
        if param != "cfg_path":  # Skip non-hyperparameters
            try:
                ax = plt.figure(figsize=(10, 6))
                data = analysis.dataframe(metric="valid_loss", mode="min")
                if param in data.columns:
                    plt.scatter(data[param], data["valid_loss"])
                    plt.xlabel(param)
                    plt.ylabel("Valid Loss")
                    plt.title(f"Impact of {param} on Validation Loss")
                    plt.savefig(os.path.join(
                        plots_dir, f"param_impact_{param}.png"))
                    plt.close()
            except Exception as e:
                print(f"Could not plot parameter {param}: {e}")

    # 3. Plot parallel coordinates plot for the top trials
    try:
        # Get the top 10 trials
        top_trials = analysis.dataframe().sort_values("valid_loss").head(10)

        # Parameters to include in the plot (exclude non-numeric or constants)
        plot_params = [p for p in best_config.keys()
                       if p != "cfg_path" and p != "scheduler"]

        # Create parallel coordinates plot
        ax = plt.figure(figsize=(15, 8))
        coords = analysis.dataframe()[plot_params + ["valid_loss"]].head(20)

        # Normalize each parameter to [0, 1] for plotting
        for param in plot_params:
            if param in coords.columns:
                min_val = coords[param].min()
                max_val = coords[param].max()
                if max_val > min_val:
                    coords[param] = (coords[param] - min_val) / \
                        (max_val - min_val)

        # Plot each trial as a line
        for i, (_, row) in enumerate(coords.iterrows()):
            xs = range(len(plot_params) + 1)
            ys = [row[param] for param in plot_params] + [row["valid_loss"]]
            plt.plot(xs, ys, 'o-', alpha=0.5)

        plt.xticks(range(len(plot_params) + 1), plot_params + ["valid_loss"])
        plt.title("Parallel Coordinates Plot of Top Trials")
        plt.savefig(os.path.join(plots_dir, "parallel_coordinates.png"))
        plt.close()
    except Exception as e:
        print(f"Could not create parallel coordinates plot: {e}")

    print(f"Hyperparameter tuning plots saved to {plots_dir}")
