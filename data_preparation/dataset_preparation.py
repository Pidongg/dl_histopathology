# Classes for preparing the dataset: splitting images and masks into patches and preparing bounding box labels. BCSSPreparer class and `get_train_test_val_img_lists` method are preserved
# from the original repo, but not used for my project.
# Created `separate_by_tiles_dict` method to split the dataset into train/validation/test sets according to a dictionary specifying which slides belong to which set instead.

import math
import tqdm
import os
import glob
import torch
import shutil

from data_preparation import data_utils, image_labelling, qupath_label_preparation
from skimage import io
from skimage.util import img_as_ubyte, shape
from collections import defaultdict


class DataPreparer:
    """
    This is a base class for preparing a dataset to be used by a model.

    Args:
        in_root_dir (Path): A path to the root directory of the input dataset.
        in_img_dir (str): The image directory of the input dataset relative to in_root_dir.
        prepared_root_dir (Path): A path to the root directory to which to write the prepared directory.
        prepared_img_dir (str): Image directory relative to prepared_root_dir.

    Attributes:
        in_root_dir (Path), in_img_dir (str), prepared_img_dir (str), prepared_label_dir (str)

    Methods:
        get_train_test_val_img_lists: Get image path lists for training, testing, and validation sets
            according to the provided splitting ratio.

    """

    def __init__(self, in_root_dir, in_img_dir, prepared_root_dir, prepared_img_dir, prepared_label_dir):
        super().__init__()
        if not os.path.exists(in_root_dir):
            raise Exception("Input directory does not exist")

        if not os.path.exists(os.path.join(in_root_dir, in_img_dir)):
            raise Exception("Input images directory does not exist")

        self.in_root_dir = in_root_dir
        self.in_img_dir = in_img_dir
        self.prepared_root_dir = prepared_root_dir
        self.prepared_img_dir = prepared_img_dir
        self.prepared_label_dir = prepared_label_dir

    def get_train_test_val_img_lists(self, train: float, test: float, val: float):
        """
        Splits the dataset into training, testing, and validation sets, then returns image paths grouped by set.

        Args:
            train (float): Fraction of the dataset to use for training
            test (float): Fraction to use for testing
            val (float): Fraction to use for validation

        Returns:
            img_lists (dict[list[path]]): A dictionary of image path lists with keys [train, test, val].
        """
        # validate arguments
        assert (train + test + val) == 1

        image_list = data_utils.list_files_of_a_type(
            os.path.join(self.in_root_dir, self.in_img_dir), ".png")
        image_list.sort()  # ensure consistent ordering
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

    def count_objects(self):
        """
        Print the number of images in each directory in the prepared image directory,
            along with the number of objects per image.
        """
        root_img_dir = os.path.join(self.prepared_root_dir, self.prepared_img_dir)
        root_label_dir = os.path.join(self.prepared_root_dir, self.prepared_label_dir)

        sets = [f for f in os.listdir(root_img_dir)]

        for set_name in sets:
            num_objects_per_class = defaultdict(int)

            img_dir = os.path.join(root_img_dir, set_name)
            label_dir = os.path.join(root_label_dir, set_name)

            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")

            for img_path in img_paths:
                img_name = data_utils.get_filename(img_path)
                label_path = os.path.join(label_dir, img_name + ".txt")

                bboxes, labels = image_labelling.bboxes_from_yolo_labels(
                    label_path)

                for l in labels:
                    num_objects_per_class[l] += 1

            print(
                f"Objects per class in set {set_name} : {num_objects_per_class}")
            print(f"Number of images in set {set_name}: {len(img_paths)}")
            num_objects_per_class.clear()


class BCSSPreparer(DataPreparer):
    """
    This is a class for preparing the BCSS dataset to be used by a model.

    Args:
        in_root_dir (Path)
        in_img_dir (str)
        in_mask_dir (str): The mask directory of the input dataset relative to in_root_dir.
        patch_w (int): Width to use for each image patch, in pixels
        patch_h (int): Height of each image patch in pixels
        prepared_root_dir (Path)
        prepared_img_dir (str)
        prepared_mask_dir (str): Mask directory relative to prepared_root_dir.
        prepared_label_dir (str)

    Attributes:
        in_root_dir (Path), in_img_dir (str), in_mask_dir (str), patch_w (int), patch_h (int),
        prepared_root_dir (Path), prepared_img_dir (str), prepared_mask_dir (str), prepared_label_dir (str)

    Methods:
        split_into_patches: Splits each image in a given list into patches, and writes them to a specified directory.
        bboxes_from_one_mask: Generates bboxes from one mask in a prepared directory.
        bboxes_from_all_masks: Generates bboxes from all masks in a prepared directory.
        show_masks_and_bboxes: Display masks and bboxes overlaid on images in a prepared directory.
    """

    def __init__(self, in_root_dir, in_img_dir, in_mask_dir, patch_w, patch_h,
                 prepared_root_dir, prepared_img_dir, prepared_mask_dir, prepared_label_dir):

        super().__init__(in_root_dir, in_img_dir, prepared_root_dir,
                         prepared_img_dir, prepared_label_dir)

        if not os.path.exists(os.path.join(in_root_dir, in_mask_dir)):
            raise Exception("Input mask directory does not exist")

        if len(data_utils.list_files_of_a_type(os.path.join(in_root_dir, in_img_dir), ".png")) != \
                len(data_utils.list_files_of_a_type(os.path.join(in_root_dir, in_mask_dir), ".png")):
            raise Exception("Number of images does not match number of masks")

        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_mask_dir = in_mask_dir
        self.prepared_mask_dir = prepared_mask_dir
        self.prepared_label_dir = prepared_label_dir

    def split_into_patches(self, image_list: list[os.path], set_type: str = "") -> None:
        """
        Given a list of image paths, splits each image into a number of patches of fixed size,
            finds the corresponding mask and splits it into matching patches,
            and saves these patches to the relevant output directory specified at initialisation.
        Based on https://github.com/nauyan/NucleiSegmentation.

        Args:
            image_list (list[path]): The paths of images to be split
            set_type (str): Name of set being processed e.g. "train", "test", "valid", so the patches can be written
                to the correct directory.
        """
        out_img_dir = os.path.join(
            self.prepared_root_dir, self.prepared_img_dir, set_type)
        out_mask_dir = os.path.join(
            self.prepared_root_dir, self.prepared_mask_dir, set_type)

        # create directories for prepared dataset if not already existing
        if not os.path.exists(out_mask_dir):
            os.makedirs(out_mask_dir)

        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        for i in tqdm.tqdm(range(len(image_list))):
            img_path = image_list[i]

            # get image name and read image
            image_name = data_utils.get_filename(img_path)
            image = io.imread(img_path)

            # based on image name, get path to matching mask and read mask
            mask_path = glob.glob(os.path.join(
                self.in_root_dir, self.in_mask_dir, image_name) + "*")[0]
            mask = io.imread(mask_path)

            # divide image into patches
            img_patches = shape.view_as_windows(image,
                                          (self.patch_w, self.patch_h, 3),
                                          (3 * self.patch_w//4, 3 * self.patch_h//4, 3))  # overlap of 25%
            img_patches = img_patches.reshape(-1,
                                              self.patch_w, self.patch_h, 3)

            mask_patches = shape.view_as_windows(mask,
                                           (self.patch_w, self.patch_h),
                                           (3 * self.patch_w//4, 3 * self.patch_h//4))
            mask_patches = mask_patches.reshape(-1, self.patch_w, self.patch_h)

            for j in range(len(img_patches)):
                # save patches to directory
                target_img_path = os.path.join(
                    out_img_dir, image_name) + "_" + str(j) + ".png"
                target_mask_path = os.path.join(
                    out_mask_dir, image_name) + "_" + str(j) + ".png"

                io.imsave(target_img_path, img_as_ubyte(img_patches[j]))
                # suppress contrast warnings
                io.imsave(target_mask_path,
                          mask_patches[j], check_contrast=False)

    def bboxes_from_one_mask(self, filename: str, set_type: str, yolo: bool = False) -> [torch.Tensor, torch.Tensor]:
        """
        A wrapper around image_labelling.bboxes_from_one_mask, extracting labels from one prepared mask in `set_type`.

        Args:
            filename (str): Filename of the image to which the mask corresponds, e.g. "image_name.png".
            set_type (str): Name of set (e.g. "train", "test", "valid").
                Label files will be output to the location out_dir/set_type.
            yolo (bool): True if YOLO-format label files are to be output.
        Returns:
            bboxes (Tensor[N, 4]): N bounding boxes (x_min, y_min, x_max, y_max), for each of N classes present.
            labels (Tensor[N]): Class index of each bounding box.
        """
        mask_path = os.path.join(
            self.prepared_root_dir, self.prepared_mask_dir, filename)

        if not os.path.exists(mask_path):
            raise Exception(
                "Provided mask file does not exist in the prepared directory.")

        out_dir = os.path.join(self.prepared_root_dir,
                               self.prepared_label_dir), set_type
        bboxes, labels, _ = image_labelling.bboxes_from_one_mask(mask_path=mask_path,
                                                                 out_dir=out_dir,
                                                                 yolo=yolo)

        return bboxes, labels

    def bboxes_from_all_masks(self, set_type: str, yolo=False):
        """
        A wrapper around image_labelling.bboxes_from_multiple_masks, extracting labels from all prepared masks in
            `set_type`.

        Args:
            set_type (str): Name of set (e.g. "train", "test", "valid").
                Label files will be output to the location out_dir/set_type.
            yolo (bool): True if YOLO-format label files are to be output.
        """
        mask_dir_path = os.path.join(
            self.prepared_root_dir, self.prepared_mask_dir, set_type)

        # check that there exist masks to extract labels from
        if len(data_utils.list_files_of_a_type(mask_dir_path, ".png")) == 0:
            raise Exception(f"The prepared mask directory {self.prepared_root_dir}/{self.prepared_mask_dir}/{set_type}"
                            f"currently contains no masks.")

        out_dir = os.path.join(self.prepared_root_dir,
                               self.prepared_label_dir, set_type)
        image_labelling.bboxes_from_multiple_masks(mask_dir_path=mask_dir_path,
                                                   out_dir=out_dir,
                                                   yolo=yolo)

    def show_masks_and_bboxes(self, set_type: str):
        """
        Displays masks and bboxes from a prepared set, one image at a time.

        Args:
            set_type (str): Name of set to view masks and bboxes from (e.g. "train", "test", "valid").
        """
        img_paths = data_utils.list_files_of_a_type(os.path.join(self.prepared_root_dir,
                                                                 self.prepared_img_dir,
                                                                 set_type),
                                                    ".png")

        for img_path in img_paths:
            filename = data_utils.get_filename(img_path)
            print("viewing", filename)
            mask_path = os.path.join(
                self.prepared_root_dir, self.prepared_mask_dir, set_type, filename + ".png")
            image_labelling.split_and_show_masks(img_path, mask_path)

            label_path = os.path.join(
                self.prepared_root_dir, self.prepared_label_dir, set_type, filename + ".txt")
            bboxes, labels = image_labelling.bboxes_from_yolo_labels(
                label_path)
            image_labelling.show_bboxes(img_path, bboxes, labels=labels)
            _ = input("enter to continue")


class TauPreparer(DataPreparer):
    """
    This is a class for preparing the tau dataset to be used by a model.

    Args:
        in_root_dir (Path)
        in_img_dir (str)
        in_label_dir (str): The mask directory of the input dataset relative to in_root_dir.
        prepared_root_dir (str)
        prepared_img_dir (str)
        prepared_label_dir (str)

    Attributes:
        in_root_dir (Path), in_img_dir (str), in_label_dir (str), prepared_root_dir (Path), prepared_img_dir (str),
        prepared_label_dir (str)
        class_to_idx (dict[str, int]): A map of class names to indices. Hard coded to match the tau dataset.
        empty_tiles_required (dict[str, int]): A dictionary mapping slide IDs to the number of empty tiles to add.
        with_segmentation (bool): If we are also processing the slides for segmentation. For segmentation masks, we treat them the same way as corresponding images.
        preprocessed_labels (bool): If the labels have already been preprocessed, we can skip the unlabelled objects removal step.

    Methods:
        prepare_labels_for_yolo: Prepares labels from text files holding all the labels in each slide, 
            by filtering out unlabelled annotations, dividing them by tile, handling cut-off objects, and for each slide, only keep a specified number of tiles with no objects.
        train_test_val_split: Splits the dataset train/test/val sets.
        show_bboxes: Visualises the boxes on images in one of the train/test/val sets.
        count_objects: Count the number of images per set along with number of objects per class.
    """

    def __init__(self, in_root_dir, in_img_dir, in_label_dir, prepared_root_dir, prepared_img_dir, prepared_label_dir, empty_tiles_required=None, with_segmentation=False, preprocessed_labels=False):
        super().__init__(in_root_dir, in_img_dir, prepared_root_dir,
                         prepared_img_dir, prepared_label_dir)

        # validate arguments
        if not os.path.exists(os.path.join(in_root_dir, in_label_dir)):
            raise Exception("Input annotations directory does not exist")

        self.in_label_dir = in_label_dir

        self.class_to_idx = {
            'TA': 0,
            'CB': 1,
            'NFT': 2,
            'Others': 3, # Confirmed with pathologist: this class is used as a synonym for tau fragments in some slides.
            'coiled': 1, # Confirmed with pathologist: this class is used as a synonym for CB in some slides.
            'tau_fragments': 3
        }
        self.empty_tiles_required = empty_tiles_required
        self.with_segmentation = with_segmentation
        self.preprocessed_labels = preprocessed_labels 

    def prepare_labels_for_yolo(self):
        """
        Goes through all the label files in `self.in_label_dir` and removes unlabelled annotations,
        divides all annotations per tile, and removes tiles with no annotations.
        """
        print(f"Processing all regions")
        root_label_dir = os.path.join(
            self.in_root_dir, self.in_label_dir)
        root_img_dir = os.path.join(
            self.prepared_root_dir, self.prepared_img_dir)
        out_dir = os.path.join(self.prepared_root_dir, self.prepared_label_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        label_preparer = qupath_label_preparation.LabelPreparer(root_label_dir=root_label_dir,
                                       root_img_dir=root_img_dir,
                                       out_dir=out_dir,
                                       class_to_idx=self.class_to_idx,
                                       with_segmentation=self.with_segmentation,
                                       preprocessed_labels=self.preprocessed_labels)
        print("\nCreating filtered detection lists...")
        label_preparer.remove_unlabelled_objects()

        print("\nSeparating labels by tile for each slide...")
        label_preparer.separate_labels_by_tile(
            area_thresholds=[0.1, 0.225, 0.195, 0.4], length_threshold=16)

        print("\nDeleting images/labels with empty label files...")
        label_preparer.filter_files_with_no_labels(inplace=False)

        if self.empty_tiles_required is not None:
            label_preparer.add_empty_tiles(self.empty_tiles_required)

    def train_test_val_split(self, train: float, test: float, valid: float):
        """
        Split the entire dataset into train/test/val sets and move the images + labels into the relevant directories.

        Args:
            train (float): Fraction of the dataset to use for training
            test (float): Fraction to use for testing
            valid (float): Fraction to use for validation

        Returns:
            img_lists (dict[list[path]]): A dictionary of image path lists with keys [train, test, val].
        """
        for region in ['Cortical', 'BG', 'DN']:
            img_region_dir = os.path.join(
                self.in_root_dir, self.in_img_dir, region)
            slide_ids = [f for f in os.listdir(img_region_dir) if os.path.isdir(
                os.path.join(img_region_dir, f))]

            print(
                f"Moving images from {region} region into test/train/val directories...")
            for i in tqdm.tqdm(range(len(slide_ids))):
                slide_id = slide_ids[i]

                in_img_dir = os.path.join(img_region_dir, slide_id)
                in_label_dir = os.path.join(
                    self.prepared_root_dir, self.prepared_label_dir, region, slide_id)

                # check that the numbers of input images and labels for this slide match
                num_images = len(
                    data_utils.list_files_of_a_type(in_img_dir, '.png'))
                num_labels = len(
                    data_utils.list_files_of_a_type(in_label_dir, '.txt'))
                if num_images != num_labels:
                    raise Exception(f"Mismatch in number of images and labels for slide {slide_id} of region {region}:"
                                    f"{num_images} images and {num_labels} labels")

                region_preparer = DataPreparer(in_root_dir=self.in_root_dir,
                                               in_img_dir=os.path.join("images", region, slide_id))

                img_list_dict = region_preparer.get_train_test_val_img_lists(
                    train, test, valid)

                # create target directories
                target_img_dir = os.path.join(
                    self.prepared_root_dir, self.prepared_img_dir)
                if not os.path.exists(target_img_dir):
                    os.makedirs(target_img_dir)

                target_label_dir = os.path.join(
                    self.prepared_root_dir, self.prepared_label_dir)
                if not os.path.exists(target_label_dir):
                    os.makedirs(target_label_dir)

                for set_type in img_list_dict:
                    # create set directories
                    if not os.path.exists(os.path.join(target_img_dir, set_type)):
                        os.mkdir(os.path.join(target_img_dir, set_type))
                    if not os.path.exists(os.path.join(target_label_dir, set_type)):
                        os.mkdir(os.path.join(target_label_dir, set_type))

                    for img_path in img_list_dict[set_type]:
                        # get image path and move it to the target directory
                        img_name = data_utils.get_filename(img_path)
                        shutil.move(img_path, os.path.join(
                            target_img_dir, set_type, img_name + ".png"))

                        # get label path and move it to the target directory
                        label_path = os.path.join(
                            in_label_dir, img_name + ".txt")
                        shutil.move(label_path, os.path.join(
                            target_label_dir, set_type, img_name + ".txt"))

                # delete empty slide directory
                os.rmdir(in_img_dir)
                os.rmdir(in_label_dir)

    def show_bboxes(self, set_type: str):
        """
        Displays bboxes from a certain set, one image at a time.

        Args:
            set_type (str): Name of set to view masks and bboxes from (e.g. "train", "test", "valid").
        """
        img_paths = data_utils.list_files_of_a_type(os.path.join(self.prepared_root_dir,
                                                                 self.prepared_img_dir,
                                                                 set_type),
                                                     ".png")

        class_to_colour = {
            0: "green",
            1: "#19ffe8",  # cyan
            2: "#a225f5",  # purple
            3: "red"
        }

        for img_path in img_paths:
            filename = data_utils.get_filename(img_path)
            print("viewing", filename)

            label_path = os.path.join(self.prepared_root_dir, self.prepared_label_dir, set_type, filename + ".txt")
            bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path)
            colours = [class_to_colour[ci] for ci in labels]
            image_labelling.show_bboxes(img_path, bboxes, labels=labels, colours=colours)
            _ = input("enter to continue")

    def separate_by_tiles_dict(self, tiles_dict: dict):
        """
        Separates images and labels into train and validation sets according to a dictionary
        specifying which slides belong to which set. Moves entire folders instead of individual files.

        Args:
            tiles_dict (dict): A dictionary specifying which slides belong to which set.
                Format: {
                    "train": {"bg": [slide_ids], "cortical": [slide_ids], "dn": [slide_ids]},
                    "validation": {"bg": [slide_ids], "cortical": [slide_ids], "dn": [slide_ids]},
                    "test": {"bg": [slide_ids], "cortical": [slide_ids], "dn": [slide_ids]}
                }
        """
        # Create target directories if they don't exist
        for set_type in tiles_dict.keys():
            target_img_dir = os.path.join(
                self.prepared_root_dir, self.prepared_img_dir, set_type)
            target_label_dir = os.path.join(
                self.prepared_root_dir, self.prepared_label_dir, set_type)

            if not os.path.exists(target_img_dir):
                os.makedirs(target_img_dir)
            if not os.path.exists(target_label_dir):
                os.makedirs(target_label_dir)

        # For each set type (only train and validation)
        for set_type, regions in tiles_dict.items():

            print(f"\nProcessing {set_type} set...")

            # For each region (bg/cortical/dn)
            for region, slide_ids in regions.items():
                print(f"Processing {region} region...")

                # For each slide in this region
                for slide_id in slide_ids:
                    # Source directories
                    src_img_dir = os.path.join(
                        self.prepared_root_dir, self.prepared_img_dir, slide_id)
                    src_label_dir = os.path.join(
                        self.prepared_root_dir, self.prepared_label_dir, slide_id)

                    if not os.path.exists(src_img_dir):
                        print(
                            f"Warning: Image directory not found for slide {slide_id}")
                        continue
                    if not os.path.exists(src_label_dir):
                        print(
                            f"Warning: Label directory not found for slide {slide_id}")
                        continue

                    # Target directories with slide ID
                    target_slide_img_dir = os.path.join(
                        self.prepared_root_dir, self.prepared_img_dir, set_type, str(slide_id))
                    target_slide_label_dir = os.path.join(
                        self.prepared_root_dir, self.prepared_label_dir, set_type, str(slide_id))

                    # Move entire directories
                    shutil.move(src_img_dir, target_slide_img_dir)
                    shutil.move(src_label_dir, target_slide_label_dir)
