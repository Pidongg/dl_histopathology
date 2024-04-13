# written with reference to https://pytorch.org/vision/0.11/auto_examples/plot_repurposing_annotations.html

from tqdm import tqdm

import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import draw_bounding_boxes

from data_preparation.utils import *


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

    masks, obj_ids = get_masks_from_mask(mask_path)

    drawn_masks = []
    for mask in masks:
        drawn_masks.append(draw_segmentation_masks(image, mask, alpha=0.8, colors="blue"))

    print("classes: ", obj_ids)
    show_images(drawn_masks)


def show_bboxes(image_path: os.path, bboxes: torch.Tensor) -> None:
    image = read_image(image_path)

    drawn_boxes = draw_bounding_boxes(image, bboxes, colors="red")
    show_images(drawn_boxes)


def bboxes_from_one_mask(mask_path: os.path, out_dir: os.path, yolo: bool = False) -> torch.Tensor:
    """
    :param mask_path: The path to the mask to be processed.
    :param out_dir: Path to the directory to which any label files should be output (only used if yolo=True)
    :param yolo: True if YOLO-format label txt files are to be output.
    :return: a Tensor containing one bounding box (x_min, y_min, x_max, y_max) for each class present.
    """
    mask = read_image(mask_path)
    masks, obj_ids = get_masks_from_mask(mask_path)

    bboxes = masks_to_boxes(masks)

    if yolo:
        mask_name = get_filename(mask_path)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_label_path = os.path.join(out_dir, mask_name) + ".txt"

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


def bboxes_from_multiple_masks(mask_dir_path: os.path, out_dir, yolo=False):
    """
    :param mask_dir_path: Full path to the directory holding the masks.
    :param out_dir: Path to the directory to which any label files should be output (only used if yolo=True)
    :param yolo: True if YOLO-format label txt files are to be output.
    :return:
    """
    # list the masks in the given directory
    mask_list = glob.glob(mask_dir_path + "/*.png")

    for i in tqdm(range(len(mask_list))):
        mask_path = mask_list[i]
        bboxes = bboxes_from_one_mask(mask_path=mask_path, out_dir=out_dir, yolo=yolo)

        # for testing purposes- visualisation code
        # image_name = get_filename(mask_path) + '.png'
        # image_path = os.path.join(self.out_root_dir, "images", image_name)
        # split_and_show_masks(image_path, mask_path)
        # show_bboxes(image_path, bboxes)


def bboxes_from_yolo_labels(label_path: os.path) -> torch.Tensor:
    """
    :param label_path: Path to a YOLO-format label .txt file
    :return: A tensor containing bbox coordinates extracted from label_path, of shape (N, 4)
        for N = no. of bboxes, and each row holding [x_min, y_min, x_max, y_max] unnormalised.
    """
    with open(label_path, "r") as f:
        # create an empty tensor
        bboxes = torch.FloatTensor()

        labels = []

        for i, line in enumerate(f):
            line = line.strip().split(' ')
            x_centre = float(line[1])
            y_centre = float(line[2])
            width = float(line[3])
            height = float(line[4])

            x_min = (x_centre - (width / 2)) * 512
            x_max = (x_centre + (width / 2)) * 512
            y_min = (y_centre - (height / 2)) * 512
            y_max = (y_centre + (height / 2)) * 512

            new_bbox = torch.Tensor([[x_min, y_min, x_max, y_max]])

            bboxes = torch.cat([bboxes, new_bbox], axis=0)

            labels.append(line[0])

    return bboxes, labels
