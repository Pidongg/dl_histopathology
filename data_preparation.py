# written with reference to `prepare_dataset.py` at https://github.com/nauyan/NucleiSegmentation
# and https://pytorch.org/vision/0.11/auto_examples/plot_repurposing_annotations.html

from skimage.util.shape import view_as_windows
from skimage.util import img_as_ubyte

import os
import glob
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from skimage import io

import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes


def show_images(images):
    """ Takes images (singular or a list) of type Image Tensor (Torch) and plots them. """
    if not isinstance(images, list):
        images = [images]
    fix, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def get_masks_from_mask(mask_path):
    """
    Takes a path to a mask image.
    Returns a Tensor of masks for each class present, and corresponding class indices.
    """
    mask = read_image(mask_path)

    # We get the unique indices, as these would be the object ids.
    obj_ids = torch.unique(mask)

    # split the color-encoded mask into a set of boolean masks.
    masks = mask == obj_ids[:, None, None]

    return masks, obj_ids


def split_and_show_masks(image_path, mask_path):
    """ Takes an image and a mask, splits the mask by class, and displays the image with each mask overlaid. """
    image = read_image(image_path)

    masks, _ = get_masks_from_mask(mask_path)

    drawn_masks = []
    for mask in masks:
        drawn_masks.append(draw_segmentation_masks(image, mask, alpha=0.8, colors="blue"))

    show_images(drawn_masks)


def show_bboxes(image_path, bboxes):
    image = read_image(image_path)

    drawn_boxes = draw_bounding_boxes(image, bboxes, colors="red")
    show_images(drawn_boxes)


class DataPreparer:
    def __init__(self, in_root_dir, out_root_dir):
        # in_root_dir and out_root_dir should each have subdirectories 'images' and 'masks'
        self.in_root_dir = in_root_dir
        self.out_root_dir = out_root_dir

    def split_into_patches(self, img_dir: str, mask_dir: str, patch_width: int, patch_height: int):
        """
        :param img_dir: name of image directory
        :param mask_dir: name of mask directory
        :param patch_width: width of each patch in pixels
        :param patch_height: height of each patch in pixels
        :return: None
        """

        image_list = glob.glob(os.path.join(self.in_root_dir, img_dir) + "/*.png")

        out_img_dir = os.path.join(self.out_root_dir, img_dir)
        out_mask_dir = os.path.join(self.out_root_dir, mask_dir)

        # create directories for prepared dataset if not already existing
        if not os.path.exists(out_mask_dir):
            os.makedirs(out_mask_dir)

        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        for img_path in image_list:
            # get image name and read image
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            image = io.imread(img_path)

            # based on image name, get path to matching mask and read mask
            mask_path = glob.glob(os.path.join(self.in_root_dir, mask_dir, image_name) + "*")[0]
            mask = io.imread(mask_path)

            # divide image into patches
            img_patches = view_as_windows(image,
                                          (patch_width, patch_height, 3),
                                          (3 * patch_width//4, 3 * patch_height//4, 3))
            img_patches = img_patches.reshape(-1, patch_width, patch_height, 3)

            mask_patches = view_as_windows(mask,
                                           (patch_width, patch_height),
                                           (3 * patch_width//4, 3 * patch_height//4))
            mask_patches = mask_patches.reshape(-1, patch_width, patch_height)

            for i in tqdm(range(len(img_patches))):
                # save patches to directory
                target_img_path = os.path.join(out_img_dir, image_name) + "_" + str(i) + ".png"
                target_mask_path = os.path.join(out_mask_dir, image_name) + "_" + str(i) + ".png"

                io.imsave(target_img_path, img_as_ubyte(img_patches[i]))
                io.imsave(target_mask_path, mask_patches[i])

    def bboxes_from_one_mask(self, mask_path, yolo: bool = False):
        """
        :param yolo: True if YOLO-format label txt files are to be output.
        :param mask_path: The path to the mask to be processed.
        :return: a Tensor containing one bounding box (x_min, y_min, x_max, y_max) for each class present.
        """
        mask = read_image(mask_path)
        masks, obj_ids = get_masks_from_mask(mask_path)

        bboxes = masks_to_boxes(masks)

        if yolo:
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            out_label_dir = os.path.join(self.out_root_dir, "bbox_labels")

            if not os.path.exists(out_label_dir):
                os.makedirs(out_label_dir)

            out_label_path = os.path.join(out_label_dir, mask_name) + ".txt"

            img_w, img_h = mask.size(1), mask.size(2)

            with open(out_label_path, 'w') as f:
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    class_idx = obj_ids[i]
                    x_centre = (bbox[0] + bbox[2]) / (2 * img_w)
                    y_centre = (bbox[1] + bbox[3]) / (2 * img_h)
                    width = (bbox[2] - bbox[0]) / img_w
                    height = (bbox[3] - bbox[1]) / img_h
                    line = f"{class_idx} {x_centre} {y_centre} {width} {height}\n"
                    f.write(line)

        return bboxes

    def bboxes_from_multiple_masks(self, mask_dir, yolo=False):
        mask_list = glob.glob(os.path.join(self.out_root_dir, mask_dir) + "/*.png")

        for mask_path in mask_list:
            bboxes = self.bboxes_from_one_mask(mask_path, yolo=yolo)

            # for testing purposes- visualisation
            # image_name = os.path.splitext(os.path.basename(mask_path))[0] + '.png'
            # image_path = os.path.join(self.out_root_dir, "images", image_name)
            # split_and_show_masks(image_path, mask_path)
            # show_bboxes(image_path, bboxes)


