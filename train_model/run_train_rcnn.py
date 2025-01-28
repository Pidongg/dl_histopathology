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
    selected_gpus = gpus[:num_gpus]
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

def get_train_transform():
    return A.Compose([
        # Horizontal and vertical flips (matching YOLO's fliplr/flipud)
        A.HorizontalFlip(p=0.572),  # fliplr: 0.572
        A.VerticalFlip(p=0.259),    # flipud: 0.259
        A.RandomRotate90(p=0.5),
        
        # Affine transforms combining rotation, translation, scale, and shear
        A.Affine(
            # Rotation: YOLO's degrees=1.34 means ±1.34°
            rotate=(-1.34, 1.34),
            
            # Translation: YOLO's translate=0.206 means ±20.6% of image size
            translate_percent=(-0.206, 0.206),
            
            # Scale: YOLO's scale=0.264 means random uniform in [1-0.264, 1+0.264]
            scale=(0.736, 1.264),  # 1 ± 0.264
            
            # Common settings
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.8  # High probability since this combines multiple transforms
        ),

        A.CLAHE(p=1.0, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
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
    
    # Create new Sequential module
    new_layers = []
    dropout_rate = 0.5
    
    for i, layer in enumerate(layers):
        # Add dropout before the last 2 Conv2dNormActivation modules
        if isinstance(layer, torchvision.ops.misc.Conv2dNormActivation) and i >= len(layers) - 5:
            new_layers.append(torch.nn.Dropout2d(p=dropout_rate))
        new_layers.append(layer)
    
    # Replace the box head with our modified version
    model.roi_heads.box_head = torch.nn.Sequential(*new_layers)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    epoch_loss = 0
    print('Training')
    
    prog_bar = tqdm.tqdm(data_loader, total=len(data_loader))
    
    optimizer.zero_grad()
    accumulation_steps = 4  # Accumulate gradients to simulate larger batch size
    
    for i, (images, targets) in enumerate(prog_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss += loss_value

        # Normalize loss for gradient accumulation
        losses = losses / accumulation_steps
        losses.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        prog_bar.set_description(f'Loss: {loss_value:.4f}')

    return epoch_loss / len(data_loader)


def validate(model, data_loader, device):
    model.train()  # Keep in train mode to get losses
    epoch_loss = 0
    print('Validating')

    prog_bar = tqdm.tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for i, (images, targets) in enumerate(prog_bar):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value

            prog_bar.set_description(f'Loss: {loss_value:.4f}')

    return epoch_loss / len(data_loader)


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
                        
    parser.add_argument("-n", "--num_gpus",
                        help="Number of GPUs to use",
                        type=int,
                        default=1)

    args = parser.parse_args()

    # Select best GPUs and set CUDA_VISIBLE_DEVICES
    best_gpus = select_best_gpus(args.num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = best_gpus
    print(f"Using GPU(s): {best_gpus}")

    # If using multiple GPUs, wrap the model in DataParallel
    if args.num_gpus > 1:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    print("Using device:", device)

    cfg = args.cfg
    OUT_MODEL_DIR = args.model_save_dir
    model_name = args.model_name
    num_epochs = args.num_epochs

    if not os.path.exists(OUT_MODEL_DIR):
        os.makedirs(OUT_MODEL_DIR)

    # get class dictionary
    with open(cfg, "r") as stream:
        cfg_dict = yaml.safe_load(stream)

        try:
            class_dict = cfg_dict['names']
            path = cfg_dict['path']
            img_train_dir = cfg_dict['train']
            img_val_dir = cfg_dict['val']
            label_train_dir = cfg_dict['train_labels']
            label_val_dir = cfg_dict['val_labels']

        except KeyError:
            sys.exit("Provided yaml file is expected to contain a 'names' field that holds a dictionary of"
                     "class indices to names, a 'path' field indicating the dataset's root directory,"
                     "a 'train' field indicating the training image directory relative to 'path',"
                     "a 'val' field indicating the validation image directory relative to 'path',"
                     "and 'train_labels' and 'val_labels' fields.")

    num_classes = len(class_dict.keys()) + 1  # one extra for background
    img_train_dir = os.path.join(path, img_train_dir)
    img_val_dir = os.path.join(path, img_val_dir)
    label_train_dir = os.path.join(path, label_train_dir)
    label_val_dir = os.path.join(path, label_val_dir)

    # use our dataset and defined transformations
    dataset = RCNNDataset(img_dir=img_train_dir,
                          label_dir=label_train_dir,
                          width=512,
                          height=512,
                          transforms=get_train_transform(),
                          device=device)

    dataset_valid = RCNNDataset(img_dir=img_val_dir,
                                label_dir=label_val_dir,
                                width=512,
                                height=512,
                                transforms=get_valid_transform(),
                                device=device)
    
    print(f"\nDataset information:")
    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(dataset_valid)}")

    def get_num_workers():
        """Get appropriate number of workers based on system"""
        num_cpus = os.cpu_count()
        if num_cpus is None:
            return 0
        # Usually 0 workers for CPU training, num_cpus for GPU
        return min(4, num_cpus) if torch.cuda.is_available() else 0

    # Adjust batch size based on number of GPUs
    base_batch_size = 4
    batch_size = base_batch_size * args.num_gpus

    # Update data loaders with new batch size
    data_loader_train = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_num_workers(),
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True
    )

    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_num_workers(),
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    print(model.roi_heads.box_head)
    
    # Wrap model in DataParallel if using multiple GPUs
    if args.num_gpus > 1 and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # save the state dict of the model with the lowest loss on the validation step
    best_valid_loss = float('inf')
    best_model_state_dict = None

    # save train and valid losses at each step
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch + 1} of {num_epochs}")
        
        # train for one epoch
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device=device)
        print(f"Training loss: {train_loss}")
        train_losses.append(train_loss)

        # update the learning rate
        lr_scheduler.step()

        # save the model
        model_path = get_unused_filename(OUT_MODEL_DIR, model_name, ".pth")
        torch.save(model.state_dict(), model_path)

        # evaluate on the validation dataset
        valid_loss = validate(model, data_loader_valid, device=device)
        print(f"Validation loss: {valid_loss}")
        valid_losses.append(valid_loss)

        if not best_model_state_dict or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state_dict = model.state_dict()

    # after all epochs, save the model with the best performance
    model_path = get_unused_filename(OUT_MODEL_DIR, "best_model", ".pth")
    torch.save(best_model_state_dict, model_path)
    # TODO: visualize the training and validation losses, and make it per class
