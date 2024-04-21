# Classes to load data for training with PyTorch's Faster RCNN implementation.
# Written with reference to this tutorial: https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

import glob
import torch
import os
import cv2
import numpy as np

from data_preparation import image_labelling
from data_preparation import utils


class Amgad2019Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, width, height, transforms=None):
        self.transforms = transforms
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.height = height
        self.width = width

        # get all the image paths in sorted order
        self.image_paths = utils.list_files_of_a_type(img_dir, ".png")
        self.all_images = [os.path.basename(image_path) for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.img_dir, image_name)

        # read the image
        image = cv2.imread(image_path)

        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0  # scale colour channels to 0-1

        # get the image's mask, which has the same name
        mask_path = os.path.join(self.mask_dir, image_name)

        # box coordinates, labels, and per-class masks are extracted from the mask
        # output_dir is empty since no labels will be written out
        boxes, labels, masks = image_labelling.bboxes_from_one_mask(mask_path=mask_path, out_dir="", yolo=False)

        # increment the labels since 0 is reserved for the background class.
        labels += 1

        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx]),
            "masks": masks
        }

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)
