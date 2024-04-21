# Classes for preparing the dataset: splitting images and masks into patches,
#   dividing the dataset into train, test and validation sets,
#   and some wrappers for extracting bbox labels from masks.

# Note: `Amgad2019_Preparer.split_into_patches(...)` was written with reference to `prepare_dataset.py`
#   at https://github.com/nauyan/NucleiSegmentation.

from skimage.util.shape import view_as_windows
from skimage.util import img_as_ubyte
from skimage import io

import math
import tqdm
import os
import glob
import torch
import shutil
import csv

from data_preparation import utils, image_labelling


class DataPreparer:
    def __init__(self,
                 in_root_dir,
                 in_img_dir):
        super().__init__()
        # validate arguments
        if not os.path.exists(in_root_dir) or \
                not os.path.exists(os.path.join(in_root_dir, in_img_dir)):
            raise Exception("Input dataset does not exist")

        self.in_root_dir = in_root_dir  # directory of the unprocessed dataset
        self.in_img_dir = in_img_dir  # directory of images relative to in_root_dir

    def get_train_test_val_img_lists(self, train: float, test: float, val: float):
        """
        Splits the dataset into training, testing, and validation sets, then returns image paths grouped by set.
        :param train: Percentage of the dataset to use for training
        :param test: Percentage to use for testing
        :param val: Percentage to use for validation
        :return: A dictionary of lists of image paths with the keys [train, test, val].
        """
        assert (train + test + val) == 1

        image_list = glob.glob(
            os.path.join(self.in_root_dir, self.in_img_dir, "*.png"))  # alternative: directly do `+ "/*.png"`
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


class DeepLesion(DataPreparer):
    def __init__(self, in_root_dir, in_img_dir,
                 prepared_root_dir, prepared_img_dir, prepared_label_dir,
                 label_file):
        super().__init__(in_root_dir, in_img_dir)
        self.prepared_root_dir = prepared_root_dir
        self.prepared_img_dir = prepared_img_dir
        self.prepared_label_dir = prepared_label_dir
        self.label_file = label_file

    def train_test_val_split(self, train: float, test: float, val: float):
        img_lists = self.get_train_test_val_img_lists(train, test, val)

        # create target directories
        target_img_dir = os.path.join(self.prepared_root_dir, self.prepared_img_dir)
        if not os.path.exists(target_img_dir):
            os.makedirs(target_img_dir)

        target_label_dir = os.path.join(self.prepared_root_dir, self.prepared_label_dir)
        if not os.path.exists(target_label_dir):
            os.makedirs(target_label_dir)

        for set_type in img_lists:
            os.mkdir(os.path.join(target_img_dir, set_type))
            for i in tqdm.tqdm(range(len(img_lists[set_type]))):
                img_path = img_lists[set_type][i]
                img_name = os.path.basename(img_path)
                shutil.move(img_path, os.path.join(target_img_dir, set_type, img_name))

    def process_extracted_images(self, in_dir):
        """
        DELETE LATER
        Delete slides which do not have a corresponding label in any of the train, test, or valid sets,
        and move them to prepared_root_dir/prepared_img_dir with the full name.
        """
        for img_dir in os.listdir(in_dir):  # walk through the top-level directory
            img_dir = os.path.join(in_dir, img_dir)  # convert to full path

            if not os.path.isdir(img_dir):  # skip non-directories
                continue

            # get just the name of the subdirectory
            dir_name = os.path.basename(img_dir)

            img_paths = utils.list_files_of_a_type(img_dir, ".png")

            for i in tqdm.tqdm(range(len(img_paths))):
                img_path = img_paths[i]
                img_name = dir_name + '_' + utils.get_filename(img_path)  # name that the label file will have

                label_found = False

                for set_type in ["train", "test", "valid"]:
                    label_path = os.path.join(self.prepared_root_dir, self.prepared_label_dir, set_type, img_name + ".txt")
                    if os.path.exists(label_path):
                        label_found = True
                        break

                if not label_found:
                    os.remove(img_path)
                else:
                    target_img_path = os.path.join(self.prepared_root_dir, self.prepared_img_dir, img_name + ".png")
                    shutil.move(img_path, target_img_path)

            os.rmdir(img_dir)

    def create_labels(self):
        label_file = os.path.join(self.in_root_dir, self.label_file)

        with open(label_file, "r") as f:
            file_reader = csv.reader(f)
            if not os.path.exists(os.path.join(self.prepared_root_dir, self.prepared_label_dir)):
                os.makedirs(os.path.join(self.prepared_root_dir, self.prepared_label_dir))
            next(file_reader, None)  # skip the headers

            for row in file_reader:
                # exclude -1 labels.
                if row[9] == "-1":
                    continue

                img_name = row[0].split('.')[0]
                target_label_path = os.path.join(self.prepared_root_dir, self.prepared_label_dir, img_name + ".txt")
                lesion_type = int(row[9]) - 1  # subtract 1 so indices start from 0

                image_size = int(row[13].split(',')[0])

                bbox = row[6].split(',')
                xmin = float(bbox[0].strip(' '))
                ymin = float(bbox[1].strip(' '))
                xmax = float(bbox[2].strip(' '))
                ymax = float(bbox[3].strip(' '))
                xc = (xmin + xmax) / (2 * image_size)
                yc = (ymin + ymax) / (2 * image_size)
                w = (xmax - xmin) / image_size
                h = (ymax - ymin) / image_size

                with open(target_label_path, "w") as label_writer:
                    label_writer.write(f"{lesion_type} {xc} {yc} {w} {h}")

        self.split_labels()

    def split_labels(self):
        for set_type in ["train", "test", "valid"]:
            label_dir = os.path.join(self.prepared_root_dir, self.prepared_label_dir, set_type)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            img_dir = os.path.join(self.prepared_root_dir, self.prepared_img_dir, set_type)
            img_list = utils.list_files_of_a_type(img_dir, ".png")
            for i in tqdm.tqdm(range(len(img_list))):
                img_path = img_list[i]
                img_name = utils.get_filename(img_path)
                label_path = os.path.join(self.prepared_root_dir, self.prepared_label_dir, img_name + ".txt")
                if os.path.exists(label_path):
                    shutil.move(label_path, os.path.join(label_dir, img_name + ".txt"))

    def show_bboxes_from_labels(self, set_type: str):
        img_paths = utils.list_files_of_a_type(os.path.join(self.prepared_root_dir,
                                                            self.prepared_img_dir,
                                                            set_type),
                                               ".png")
        for img_path in img_paths:
            filename = utils.get_filename(img_path)
            print("viewing", filename)

            label_path = os.path.join(self.prepared_root_dir, self.prepared_label_dir, set_type, filename + ".txt")
            bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path)
            print(bboxes)
            image_labelling.show_bboxes(img_path, bboxes, labels=labels)
            _ = input("enter to continue")


class Amgad2019_Preparer(DataPreparer):
    def __init__(self,
                 in_root_dir, prepared_root_dir,
                 patch_w, patch_h,
                 in_img_dir, in_mask_dir,
                 prepared_img_dir, prepared_mask_dir):
        super().__init__(in_root_dir, in_img_dir)

        # validate arguments
        if not os.path.exists(os.path.join(in_root_dir, in_mask_dir)):
            raise Exception("Input dataset does not exist")

        if len(utils.list_files_of_a_type(os.path.join(in_root_dir, in_img_dir), ".png")) != \
                len(utils.list_files_of_a_type(os.path.join(in_root_dir, in_mask_dir), ".png")):
            raise Exception("Number of images does not match number of masks")

        self.prepared_root_dir = prepared_root_dir  # directory of the prepared dataset
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_mask_dir = in_mask_dir  # directory of masks relative to in_root_dir
        self.prepared_img_dir = prepared_img_dir  # directory of images relative to prepared_root_dir
        self.prepared_mask_dir = prepared_mask_dir  # directory of masks relative to prepared_root_dir

        self.label_dir = "labels"

    def split_into_patches(self, image_list: list[os.path], set_type: str) -> None:
        """
        Given a list of image paths, splits each image into a number of patches of fixed size,
            finds the corresponding mask and splits it into matching patches,
            and saves these patches to the relevant output directory specified at initialisation.

        :param image_list: list of paths to images to be split
        :param set_type: type of set being processed e.g. "train", "test", "valid"
        :return: None
        """
        out_img_dir = os.path.join(self.prepared_root_dir, self.prepared_img_dir, set_type)
        out_mask_dir = os.path.join(self.prepared_root_dir, self.prepared_mask_dir, set_type)

        # create directories for prepared dataset if not already existing
        if not os.path.exists(out_mask_dir):
            os.makedirs(out_mask_dir)

        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        for i in tqdm.tqdm(range(len(image_list))):
            img_path = image_list[i]

            # get image name and read image
            image_name = utils.get_filename(img_path)
            image = io.imread(img_path)

            # based on image name, get path to matching mask and read mask
            mask_path = glob.glob(os.path.join(self.in_root_dir, self.in_mask_dir, image_name) + "*")[0]
            mask = io.imread(mask_path)

            # divide image into patches
            img_patches = view_as_windows(image,
                                          (self.patch_w, self.patch_h, 3),
                                          (3 * self.patch_w//4, 3 * self.patch_h//4, 3))  # overlap of 25%
            img_patches = img_patches.reshape(-1, self.patch_w, self.patch_h, 3)

            mask_patches = view_as_windows(mask,
                                           (self.patch_w, self.patch_h),
                                           (3 * self.patch_w//4, 3 * self.patch_h//4))
            mask_patches = mask_patches.reshape(-1, self.patch_w, self.patch_h)

            for j in range(len(img_patches)):
                # save patches to directory
                target_img_path = os.path.join(out_img_dir, image_name) + "_" + str(j) + ".png"
                target_mask_path = os.path.join(out_mask_dir, image_name) + "_" + str(j) + ".png"

                io.imsave(target_img_path, img_as_ubyte(img_patches[j]))
                io.imsave(target_mask_path, mask_patches[j], check_contrast=False)  # suppress contrast warnings

    def bboxes_from_one_mask(self, filename: str, set_type: str, yolo: bool = False) -> [torch.Tensor, torch.Tensor]:
        """
        A wrapper around image_labelling.bboxes_from_one_mask(...).
        :param filename: Filename of the image to which the mask corresponds, e.g. "image_name.png".
        :param yolo: True if YOLO-format label txt files are to be output.
        :param set_type: Name of set (e.g. "train", "test", "valid").
            Label files will be output to the location out_dir/set_type.
        :return: a Tensor containing one bounding box (x_min, y_min, x_max, y_max) for each class present.
        """
        mask_path = os.path.join(self.prepared_root_dir, self.prepared_mask_dir, filename)
        out_dir = os.path.join(self.prepared_root_dir, "labels"), set_type
        bboxes, labels, _ = image_labelling.bboxes_from_one_mask(mask_path=mask_path,
                                                      out_dir=out_dir,
                                                      yolo=yolo)

        return bboxes, labels

    def bboxes_from_all_masks(self, set_type: str, yolo=False):
        """
        A wrapper around image_labelling.bboxes_from_multiple_masks(...).
        :param set_type: Name of set (e.g. "train", "test", "valid").
            Label files will be output to the location out_dir/set_type.
        :param yolo:
        :return: None
        """
        mask_dir_path = os.path.join(self.prepared_root_dir, self.prepared_mask_dir, set_type)
        out_dir = os.path.join(self.prepared_root_dir, "labels", set_type)
        image_labelling.bboxes_from_multiple_masks(mask_dir_path=mask_dir_path,
                                                   out_dir=out_dir,
                                                   yolo=yolo)

    def show_masks_and_bboxes(self, set_type: str):
        """
        Displays masks and bboxes from a certain set, one image at a time.
        :param set_type: Name of set to view masks and bboxes from (e.g. "train", "test", "valid").
        :return:
        """
        img_paths = utils.list_files_of_a_type(os.path.join(self.prepared_root_dir,
                                                            self.prepared_img_dir,
                                                            set_type),
                                               ".png")

        for img_path in img_paths:
            filename = utils.get_filename(img_path)
            print("viewing", filename)
            mask_path = os.path.join(self.prepared_root_dir, self.prepared_mask_dir, set_type, filename + ".png")
            image_labelling.split_and_show_masks(img_path, mask_path)

            label_path = os.path.join(self.prepared_root_dir, self.label_dir, set_type, filename + ".txt")
            bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path)
            image_labelling.show_bboxes(img_path, bboxes, labels=labels)
            _ = input("enter to continue")
