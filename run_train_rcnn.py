import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
import tqdm
import glob
import argparse
import yaml
import sys

from data_preparation.rcnn_datasets import RCNNDataset


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
        last_used_filename = os.path.splitext(os.path.basename(last_used_path))[0]
        if last_used_filename == filename:
            path_to_use = f"{out_dir}/{filename}_0{extension}"
        else:
            idx = int(last_used_filename.split('_')[-1]) + 1
            path_to_use = f"{out_dir}/{filename}_{idx}{extension}"

    return path_to_use


# define the training transforms
def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one with the required number of classes
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

        loss_dict = model(images, targets)  # r-cnn gives back loss values?
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
#     model.eval() # commenting this out so we can get the loss
    epoch_loss = 0
    print('Validating')

    # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for i, (images, targets) in enumerate(prog_bar):
            optimizer.zero_grad()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

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

    args = parser.parse_args()

    cfg = args.cfg
    OUT_MODEL_DIR = args.model_save_dir
    # OUT_MODEL_DIR = "./models/RCNN/Tau"
    model_name = args.model_name
    # model_name = "rcnn_tau"
    num_epochs = args.num_epochs

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("using device: ", device)

    # specify dirs to save models and plots to
    # OUT_MODEL_DIR = "./rcnn/models"
    # OUT_PLOTS_DIR = "./rcnn/loss_plots"

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

    # img_train_dir = "./prepared_data/Tau/images/train"
    # img_val_dir = "./prepared_data/Tau/images/valid"
    # label_train_dir = "./prepared_data/Tau/labels/train"
    # label_val_dir = "./prepared_data/Tau/labels/valid"

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
                                transforms=get_train_transform(),
                                device=device)

    # define training and validation data loaders
    data_loader_train = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # model.load_state_dict(torch.load("models/RCNN/Tau/rcnn_tau_4.pth"))

    # move model to the right device
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

    # save the state dict of the model with the lowest loss on the validation step throughout training
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
        # get a file path not already in use that also uses the filename
        model_path = get_unused_filename(OUT_MODEL_DIR, model_name, ".pth")

        torch.save(model.state_dict(), model_path)

        # evaluate on the validation dataset
        valid_loss = validate(model, data_loader_valid, device=device)
        print(f"Validation loss: {valid_loss}")
        valid_losses.append(valid_loss)

        if not best_model_state_dict or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state_dict = model.state_dict()

    # display and save loss plots
    # figure_1, train_ax = plt.subplots()
    # figure_2, valid_ax = plt.subplots()
    #
    # train_ax.plot(train_losses, color='blue')
    # train_ax.set_xlabel('iterations')
    # train_ax.set_ylabel('train loss')
    #
    # valid_ax.plot(valid_losses, color='red')
    # valid_ax.set_xlabel('iterations')
    # valid_ax.set_ylabel('validation loss')
    #
    # train_fig_out_path = get_unused_filename(OUT_PLOTS_DIR, "train_loss", ".png")
    # valid_fig_out_path = get_unused_filename(OUT_PLOTS_DIR, "valid_loss", ".png")
    #
    # figure_1.savefig(train_fig_out_path)
    # figure_2.savefig(valid_fig_out_path)
    # print('SAVING PLOTS COMPLETE...')
    #
    # # after all epochs have been completed, save the model with the best performance
    # model_name = "best_model"

    # get a file path not already in use that also uses the filename
    model_path = get_unused_filename(OUT_MODEL_DIR, model_name, ".pth")

    torch.save(best_model_state_dict, model_path)

