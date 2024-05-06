# written with reference to `prepare_dataset.py` at https://github.com/nauyan/NucleiSegmentation
# and https://pytorch.org/vision/0.11/auto_examples/plot_repurposing_annotations.html

from skimage.util.shape import view_as_windows
from skimage.util import img_as_ubyte

import math

from tqdm import tqdm

from skimage import io

import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import draw_bounding_boxes

from data_utils import *


def get_masks_from_mask(mask_path: os.path):
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


def split_and_show_masks(image_path: os.path, mask_path: os.path) -> None:
    """ Takes an image and a mask, splits the mask by class, and displays the image with each mask overlaid. """
    image = read_image(image_path)

    masks, _ = get_masks_from_mask(mask_path)

    drawn_masks = []
    for mask in masks:
        drawn_masks.append(draw_segmentation_masks(image, mask, alpha=0.8, colors="blue"))

    show_images(drawn_masks)


def show_bboxes(image_path: os.path, bboxes: torch.Tensor) -> None:
    image = read_image(image_path)

    drawn_boxes = draw_bounding_boxes(image, bboxes, colors="red")
    show_images(drawn_boxes)





class DataPreparer:
    def __init__(self, in_root_dir, out_root_dir):
        # in_root_dir and out_root_dir should each have subdirectories 'images' and 'masks'
        self.in_root_dir = in_root_dir  # directory of the unprocessed dataset
        self.out_root_dir = out_root_dir  # directory of the prepared dataset

    def split_into_patches(self, img_dir: str, mask_dir: str, patch_width: int, patch_height: int,
                           image_list: list[os.path]) -> None:
        """
        :param img_dir: name of image directory
        :param mask_dir: name of mask directory
        :param patch_width: width of each patch in pixels
        :param patch_height: height of each patch in pixels
        :param image_list: list of paths to images to be split
        :return: None
        """
        out_img_dir = os.path.join(self.out_root_dir, img_dir)
        out_mask_dir = os.path.join(self.out_root_dir, mask_dir)

        # create directories for prepared dataset if not already existing
        if not os.path.exists(out_mask_dir):
            os.makedirs(out_mask_dir)

        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        for i in tqdm(range(len(image_list))):
            img_path = image_list[i]
            # get image name and read image
            image_name = get_filename(img_path)
            image = io.imread(img_path)

            # based on image name, get path to matching mask and read it
            mask_path = glob.glob(os.path.join(self.in_root_dir, mask_dir, image_name) + "*")[0]
            mask = io.imread(mask_path)

            # divide image into patches
            img_patches = view_as_windows(image,
                                          (patch_width, patch_height, 3),
                                          (3 * patch_width//4, 3 * patch_height//4, 3))  # overlap of 25%
            img_patches = img_patches.reshape(-1, patch_width, patch_height, 3)

            mask_patches = view_as_windows(mask,
                                           (patch_width, patch_height),
                                           (3 * patch_width//4, 3 * patch_height//4))
            mask_patches = mask_patches.reshape(-1, patch_width, patch_height)

            for j in range(len(img_patches)):
                # save patches to directory
                target_img_path = os.path.join(out_img_dir, image_name) + "_" + str(j) + ".png"
                target_mask_path = os.path.join(out_mask_dir, image_name) + "_" + str(j) + ".png"

                io.imsave(target_img_path, img_as_ubyte(img_patches[j]))
                io.imsave(target_mask_path, mask_patches[j], check_contrast=False)  # suppress contrast warnings

    def bboxes_from_one_mask(self, mask_path: os.path, set_type: str, yolo: bool = False) -> torch.Tensor:
        """
        :param yolo: True if YOLO-format label txt files are to be output.
        :param mask_path: The path to the mask to be processed.
        :param set_type: "train", "test", or "val".
        :return: a Tensor containing one bounding box (x_min, y_min, x_max, y_max) for each class present.
        """
        mask = read_image(mask_path)
        masks, obj_ids = get_masks_from_mask(mask_path)

        bboxes = masks_to_boxes(masks)

        if yolo:
            mask_name = get_filename(mask_path)
            out_label_dir = os.path.join(self.out_root_dir, "labels", set_type)

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

    def bboxes_from_multiple_masks(self, mask_dir, set_type, yolo=False):
        mask_list = glob.glob(os.path.join(self.out_root_dir, mask_dir) + "/*.png")

        for i in tqdm(range(len(mask_list))):
            mask_path = mask_list[i]
            bboxes = self.bboxes_from_one_mask(mask_path=mask_path, set_type=set_type, yolo=yolo)

            # for testing purposes- visualisation code
            # image_name = get_filename(mask_path) + '.png'
            # image_path = os.path.join(self.out_root_dir, "images", image_name)
            # split_and_show_masks(image_path, mask_path)
            # show_bboxes(image_path, bboxes)




    def train_test_val_split(self, train: float, test: float, val: float, img_dir: str) -> dict[str, list[os.path]]:
        """
        :param train: Percentage of the dataset to use for training
        :param test: Percentage to use for testing
        :param val: Percentage to use for validation
        :param img_dir: Name of directory which stores images
        :return: After splitting the dataset into training, testing, and validation, returns a dictionary of lists of
        image paths with the keys [train, test, val].
        """
        assert (train + test + val) == 1

        image_list = glob.glob(os.path.join(self.in_root_dir, img_dir) + "/*.png")
        image_list.sort()  # ensure the same order each time
        n = len(image_list)
        train_n = math.ceil(n * 0.8)
        test_n = math.floor(n * 0.1)

        train_img_list = image_list[:train_n]
        test_img_list = image_list[train_n:train_n + test_n]
        valid_img_list = image_list[train_n + test_n:]

        return {
            "train": train_img_list,
            "test": test_img_list,
            "valid": valid_img_list
        }