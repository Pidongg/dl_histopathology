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

        # get all the image paths recursively
        self.image_paths = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(root, file))
        
        # Sort for consistency
        self.image_paths = sorted(self.image_paths)
        
        # Get relative paths for matching with labels
        self.all_images = [os.path.relpath(path, img_dir) for path in self.image_paths]
        
        print(f"Found {len(self.image_paths)} images in {img_dir}")

    def __getitem__(self, idx):
        try:
            # get the image path
            image_path = self.image_paths[idx]
            
            # get the relative path for matching with label
            rel_path = self.all_images[idx]
            image_name = os.path.splitext(rel_path)[0]

            # read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            # convert BGR to RGB color format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0  # scale colour channels to 0-1

            # get the corresponding label path
            label_path = os.path.join(self.label_dir, image_name + ".txt")
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            # box coordinates and labels are extracted from the corresponding yolo file
            boxes, labels = image_labelling.bboxes_from_yolo_labels(label_path, normalised=False)
            
            # Convert to numpy arrays
            boxes = np.array(boxes)
            labels = np.array(labels, dtype=int)

            # apply the image transforms BEFORE incrementing labels
            if self.transforms:
                try:
                    sample = self.transforms(image=image,
                                          bboxes=boxes,
                                          labels=labels)
                    image = sample['image']
                    boxes = torch.Tensor(sample['bboxes'])
                    labels = torch.tensor(sample['labels'])
                except ValueError as e:
                    print(f"Transform error for {image_path}: {e}")
                    print("Continuing without transforms")
                    
                    transforms = []
                    transforms.append(T.ToDtype(torch.float, scale=True))
                    transforms.append(T.ToPureTensor())
                    transforms = T.Compose(transforms)
                    image = transforms(image)

            # NOW increment the labels for RCNN (after transforms)
            labels = labels + 1

            # area of the bounding boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # no crowd instances
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            # prepare the final `target` dictionary
            target = {
                "boxes": boxes,
                "labels": labels,
                "area": area,
                "iscrowd": iscrowd,
                "image_id": torch.tensor([idx])
            }

            # Move everything to device and ensure correct types
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32).to(self.device)
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64).to(self.device)
            target['area'] = torch.as_tensor(target['area'], dtype=torch.float32).to(self.device)
            target['iscrowd'] = target['iscrowd'].to(self.device)
            target['image_id'] = target['image_id'].to(self.device)

            return image, target

        except Exception as e:
            print(f"Error processing {image_path} with label {label_path}: {str(e)}")
            raise

    def __len__(self):
        return len(self.image_paths)