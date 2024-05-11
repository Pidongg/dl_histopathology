# Classes to load data for training with PyTorch's Faster RCNN implementation.
# Written with reference to this tutorial: https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

import torch
import os
import cv2
import numpy as np

from . import image_labelling, data_utils

from torchvision.transforms import v2 as T


class RCNNDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, width, height, device, transforms=None):
        self.transforms = transforms
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.height = height
        self.width = width
        self.device = device

        # get all the image paths in sorted order
        self.image_paths = data_utils.list_files_of_a_type(img_dir, ".png")
        self.all_images = [os.path.basename(image_path) for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = os.path.splitext(self.all_images[idx])[0]
        image_path = os.path.join(self.img_dir, image_name + '.png')

        # read the image
        image = cv2.imread(image_path)

        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # scale colour channels to 0-1

        # get the image's label file, which has the same name
        label_path = os.path.join(self.label_dir, image_name + ".txt")

        # box coordinates and labels are extracted from the corresponding yolo file
        # output_dir is empty since no labels will be written out
        boxes, labels = image_labelling.bboxes_from_yolo_labels(label_path, normalised=False)
        labels = torch.Tensor(labels).to(self.device)

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
            "image_id": torch.tensor([idx])
        }

        # apply the image transforms
        if self.transforms:
            try:
                sample = self.transforms(image=image,
                                         bboxes=target['boxes'].cpu(),  # since this gets converted to numpy
                                         labels=labels.cpu())
                image = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
            except ValueError as e:
                # continue without applying any transforms - limited precision sometimes causes problems
                print(e)
                print("Continuing without applying transforms")

                transforms = []
                transforms.append(T.ToDtype(torch.float, scale=True))
                transforms.append(T.ToPureTensor())
                transforms = T.Compose(transforms)
                image = transforms(image)

        return image, target

    def __len__(self):
        return len(self.all_images)
