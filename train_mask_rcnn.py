import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from data_preparation import utils
from torch.utils.data import DataLoader
import tqdm
import glob
import matplotlib.pyplot as plt

from data_preparation.rcnn_datasets import Amgad2019Dataset

from torchvision_references.engine import train_one_epoch, evaluate


def get_unused_filename(out_dir, filename, extension):
    """
    Given an output directory and desired filename and extension, return a path in the directory
    that uses the filename + an index not already in use.

    Extension should be given with a '.', e.g. extension=".png".
    """
    matching_paths = glob.glob(f"{out_dir}/{filename}*{extension}")
    matching_paths.sort()
    if len(matching_paths) == 0:
        path_to_use = f"{out_dir}/model.pth"
    else:
        last_used_path = matching_paths[-1]
        last_used_filename = utils.get_filename(last_used_path)
        if last_used_filename == filename:
            path_to_use = f"{out_dir}/{filename}_0{extension}"
        else:
            idx = int(filename.split('_')[-1]) + 1
            path_to_use = f"{out_dir}/{filename}_{idx}{extension}"

    return path_to_use


# define the training tranforms
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
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


# def train_one_epoch(model, optimizer, data_loader, device):
#     """
#     Written with reference to https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
#     Function for running training iterations
#     """
#     model.train()
#     epoch_loss = 0
#     print('Training')
#
#     # initialize tqdm progress bar
#     prog_bar = tqdm.tqdm(data_loader, total=len(data_loader))
#
#     for i, (images, targets) in enumerate(prog_bar):
#         optimizer.zero_grad()
#
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         loss_dict = model(images, targets)  # r-cnn gives back loss values?
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#         epoch_loss += loss_value
#
#         losses.backward()
#         optimizer.step()
#
#         # update the loss value beside the progress bar for each iteration
#         prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
#
#     return epoch_loss / len(data_loader)

def validate(model, data_loader, device):
    """
    Function for evaluating performance on the validation set at each epoch
    """
    model.eval()
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
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # specify dirs to save models and plots to
    OUT_MODEL_DIR = "./rcnn/models"
    OUT_PLOTS_DIR = "./rcnn/loss_plots"

    if not os.path.exists(OUT_MODEL_DIR):
        os.makedirs(OUT_MODEL_DIR)

    num_classes = 23
    # use our dataset and defined transformations
    dataset = Amgad2019Dataset(img_dir='./prepared_data/Amgad2019/images/train',
                                mask_dir='./prepared_data/Amgad2019/masks/train',
                                width=512,
                                height=512,
                                transforms=get_train_transform())

    dataset_valid = Amgad2019Dataset(img_dir='./prepared_data/Amgad2019/images/valid',
                                mask_dir='./prepared_data/Amgad2019/masks/valid',
                                width=512,
                                height=512,
                                transforms=get_valid_transform())

    # define training and validation data loaders
    data_loader_train = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

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

    # training for `num_epochs` epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_valid, device=device)

    # # save the state dict of the model with the lowest loss on the validation step throughout training
    # best_valid_loss = float('inf')
    # best_model_state_dict = None
    #
    # # save train and valid losses at each step
    # train_losses = []
    # valid_losses = []
    #
    # for epoch in range(num_epochs):
    #     print(f"\nEPOCH {epoch+1} of {num_epochs}")
    #     # train for one epoch
    #     train_loss = train_one_epoch(model, optimizer, data_loader_train, device=device)
    #     train_losses.append(train_loss)
    #
    #     # update the learning rate
    #     lr_scheduler.step()
    #
    #     # evaluate on the validation dataset
    #     valid_loss = validate(model, data_loader_valid, device=device)
    #     valid_losses.append(valid_loss)
    #
    #     if not best_model_state_dict or valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         best_model_state_dict = model.state_dict()
    #
    #
    # # display and save loss plots
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
    # model_name = "model"
    #
    # # get a file path not already in use that also uses the filename
    # model_path = get_unused_filename(OUT_MODEL_DIR, model_name, ".pth")
    #
    # torch.save(best_model_state_dict, model_path)

