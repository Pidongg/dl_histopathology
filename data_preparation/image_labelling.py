# Utility functions to handle image masks, extract bounding boxes, and visualise masks/bboxes.
# Written with reference to https://pytorch.org/vision/0.11/auto_examples/plot_repurposing_annotations.html.

from tqdm import tqdm

import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from .data_utils import *


def get_masks_from_mask(mask_path: os.path):
    """
    Extracts binary masks from a multiclass mask.

    Args:
        mask_path (path): A path to the mask to separate.

    Returns:
        masks (Tensor[nc, w, h]): Masks for each of nc classes present in the original mask.
        obj_ids (Tensor[nc]): Class indices corresponding to each of the masks.
    """
    mask = read_image(mask_path)

    # We get the unique values in the mask, as these would be the object ids.
    obj_ids = torch.unique(mask)

    # split the color-encoded mask into a set of boolean masks.
    masks = mask == obj_ids[:, None, None]

    return masks, obj_ids


def split_and_show_masks(image_path: os.path, mask_path: os.path) -> None:
    """
    Takes an image and a mask, splits the mask by class, and displays the image with each mask overlaid.
    """
    image = read_image(image_path)

    masks, obj_ids = get_masks_from_mask(mask_path)

    drawn_masks = []
    for mask in masks:
        drawn_masks.append(draw_segmentation_masks(image, mask, alpha=0.8, colors="blue"))

    print("classes: ", obj_ids)
    show_images(drawn_masks)


def show_bboxes(image_path: os.path, bboxes: torch.Tensor, labels=None, colours="red") -> None:
    """
    Displays an image with bounding boxes overlaid.

    Takes:
        image_path (path): Path to the image to display.
        bboxes (Tensor[n, 4]): Coordinates of n bounding boxes.
        labels (list[int]): Labels of each bounding box.
        colors (list[str]): Color of each bounding box based on class.
    """
    image = read_image(image_path)

    if labels:
        labels = [str(l) for l in labels]  # convert to list[str] to work with show_bboxes

    drawn_boxes = draw_bounding_boxes(image, bboxes, colors=colours, labels=labels, width=2,
                                      font=r"C:\Windows\Fonts\Arial.ttf", font_size=30)

    show_images(drawn_boxes)


def bboxes_from_one_mask(mask_path: os.path, out_dir: os.path, yolo: bool = False):
    """
    Takes a mask and converts it to bounding boxes.

    Takes:
        mask_path (path): Path to the mask to be processed.
        out_dir (path): Directory to which any label files should be output (only used if yolo=True)
        yolo (bool): True if YOLO-format label txt files are to be output.

    Returns:
        bboxes (Tensor[n, 4]): One bounding box (x_min, y_min, x_max, y_max) for each class present in the mask
        obj_ids (Tensor[n]): Class index label for each bounding box
        masks (Tensor[n, w, h]): Binary masks corresponding to each bounding box
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

    return bboxes, obj_ids, masks


def bboxes_from_multiple_masks(mask_dir_path: os.path, out_dir, yolo=False):
    """
    Takes a directory of masks and converts them all to bounding boxes.

    Takes:
        mask_dir_path (path): Directory of masks to be processed.
        out_dir (path): Directory to which any label files should be output (only used if yolo=True)
        yolo (bool): True if YOLO-format label txt files are to be output.
    """
    mask_list = glob.glob(mask_dir_path + "/*.png")

    for i in tqdm(range(len(mask_list))):
        mask_path = mask_list[i]
        bboxes, labels, _ = bboxes_from_one_mask(mask_path=mask_path, out_dir=out_dir, yolo=yolo)

        # for testing purposes- visualisation code
        # image_name = get_filename(mask_path) + '.png'
        # image_path = os.path.join(self.out_root_dir, "images", image_name)
        # split_and_show_masks(image_path, mask_path)
        # show_bboxes(image_path, bboxes)


def bboxes_from_yolo_labels(label_path: os.path, normalised: bool = False):
    """
    Reads boxes from a txt file holding YOLO-format labels and outputs them in a tensor of
        (x_min, y_min, x_max, y_max) boxes along with class indices.

    Args:
        label_path: Path to a YOLO-format label .txt file

    Returns:
        bboxes (Tensor[N, 4]): bbox coordinates extracted from label_path, in (x_min, y_min, x_max, y_max) format.
            Coordinates are unnormalised by default.
        labels (list[int]): Class index label for each bounding box
    """
    device = (torch.device(f'cuda:{torch.cuda.current_device()}')
              if torch.cuda.is_available()
              else 'cpu')

    with open(label_path, "r") as f:
        # create an empty tensor
        bboxes = torch.FloatTensor().to(device)

        labels = []

        for i, line in enumerate(f):
            line = line.strip().split(' ')
            x_centre = float(line[1])
            y_centre = float(line[2])
            width = float(line[3])
            height = float(line[4])

            if normalised:
                x_min = (x_centre - (width / 2))
                x_max = (x_centre + (width / 2))
                y_min = (y_centre - (height / 2))
                y_max = (y_centre + (height / 2))
            else:
                x_min = (x_centre - (width / 2)) * 512
                x_max = (x_centre + (width / 2)) * 512
                y_min = (y_centre - (height / 2)) * 512
                y_max = (y_centre + (height / 2)) * 512

            new_bbox = torch.Tensor([[x_min, y_min, x_max, y_max]]).to(device)

            bboxes = torch.cat([bboxes, new_bbox], axis=0)

            labels.append(int(line[0]))

    return bboxes, labels