# Classes to read labels exported directly from QuPath and export them into YOLO format.

import os
import tqdm

from data_preparation import utils


class Label:
    """
    A class to process the output of export_detections.groovy and create label txt files
    for each of the tiles created from export_image_tiles.
    """
    class_to_idx = {
        'TA': 0,
        'CB': 1,
        'NFT': 2,
        'tau_fragments': 3,
        'non_tau': 4
    }

    def __init__(self, label: str):
        class_name, roi = label.strip().split(':')
        class_name = class_name.strip('[] ')
        if class_name in Label.class_to_idx:
            self.class_idx = Label.class_to_idx[class_name]
            obj_info = roi[roi.find('('):].strip('()').split(',')
            self.x0 = int(obj_info[0])
            self.y0 = int(obj_info[1])
            self.w = int(obj_info[2])
            self.h = int(obj_info[3])
        else:
            self.x0 = 0
            self.y0 = 0
            self.w = 0
            self.h = 0
            self.class_idx = -1


class LabelPreparer:
    def __init__(self, root_label_dir: os.path, root_img_dir: os.path, out_dir: os.path):
        """
        out_dir: The directory to which filtered label files will be output directly.

        All functions in this class depend on label files from QuPath being named "{slide_id}_detections.txt",
        and all filtered label files being named "{slide_id}_filtered_detections.txt".
        """
        self.root_label_dir = root_label_dir
        self.root_img_dir = root_img_dir
        self.out_dir = out_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def remove_unlabelled_objects(self):
        """
        Reads all the label files in `self.root_label_dir` (in the format exported directly from QuPath)
        and for each, writes a new label file containing only annotations with labels (i.e. != "unlabelled" or "others")
        to `self.out_dir`.
        """
        detection_files = utils.list_files_of_a_type(self.root_label_dir, ".txt")
        for i in tqdm.tqdm(range(len(detection_files))):
            detection_file = detection_files[i]
            print(f"\nProcessing {detection_file}...")
            split_filename = utils.get_filename(detection_file).split('_')

            if split_filename[-1] == "filtered":  # check this is an unfiltered label file
                continue

            slide_id = split_filename[0]

            with open(detection_file, 'r') as detections:
                output_path = os.path.join(self.out_dir, f"{slide_id}_detections_filtered.txt")

                with open(output_path, 'w') as filtered_detections:
                    for line in detections:
                        object_label = line.split(':')[0]
                        if object_label not in ["[Unlabelled] ", "[Others] "]:
                            filtered_detections.write(line)

    def separate_labels_by_tile(self):
        """
        Reads all the filtered label files in `self.out_dir` (i.e. no more 'Unlabelled' annotations)
        and separates the labels into one .txt file for each image tile in `self.root_img_dir`.

        Output label files are grouped into directories named for their slide_id.
        """
        detection_files = utils.list_files_of_a_type(self.out_dir, ".txt")

        for i in tqdm.tqdm(range(len(detection_files))):
            detection_file = detection_files[i]
            split_filename = utils.get_filename(detection_file).split('_')

            if split_filename[-1] != "filtered":  # check this is a filtered label file
                continue

            slide_id = split_filename[0]

            img_dir = os.path.join(self.root_img_dir, slide_id)
            out_label_dir = os.path.join(self.out_dir, slide_id)

            if not os.path.exists(out_label_dir):
                os.makedirs(out_label_dir)

            img_paths = utils.list_files_of_a_type(img_dir, ".png")

            for tile in img_paths:
                filename = utils.get_filename(tile)

                # process filename to get tile bounds (a lot of hard coded values)
                tile_info = filename.split(',')
                tile_x0 = int(tile_info[1][2:])
                tile_y0 = int(tile_info[2][2:])
                tile_w = int(tile_info[3][2:])
                tile_h = int(tile_info[4][2:-1])

                with open(detection_file, 'r') as all_anns:
                    # read through the entire filtered detection file and write only the labels
                    # that fall inside this tile to a corresponding label file.
                    all_anns.readline()  # skip first line

                    out_label_file = os.path.join(out_label_dir, f"{filename}.txt")

                    with open(out_label_file, 'w') as label_f:
                        for obj in all_anns:
                            # process the line to see if the detection is in this tile
                            label = Label(obj)

                            if label.class_idx == -1:
                                # label is not in a class of interest, so skip
                                continue

                            # check bounds
                            if label.x0 >= tile_x0 and (label.x0 + label.w) < (tile_x0 + tile_w) and \
                                    label.y0 >= tile_y0 and (label.y0 + label.h) < (tile_y0 + tile_h):

                                x_centre = (label.x0 + label.w/2 - tile_x0) / tile_w
                                y_centre = (label.y0 + label.h/2 - tile_y0) / tile_h

                                norm_width = label.w / tile_w
                                norm_height = label.h / tile_h

                                label_yolo = f"{label.class_idx} {x_centre} {y_centre} {norm_width} {norm_height}\n"

                                label_f.write(label_yolo)

    def delete_files_with_no_labels(self):
        """
        Goes through every image in `self.root_img_dir` and deletes all images with empty label files.
        """
        img_dirs = [f for f in os.listdir(self.root_img_dir) if os.path.isdir(os.path.join(self.root_img_dir, f))]

        for slide_id in img_dirs:
            print(f"\nProcessing slide {slide_id}...")
            img_dir = os.path.join(self.root_img_dir, slide_id)
            label_dir = os.path.join(self.out_dir, slide_id)

            img_paths = utils.list_files_of_a_type(img_dir, ".png")
            label_paths = utils.list_files_of_a_type(label_dir, ".txt")

            # check that every image has a label file.
            if len(img_paths) != len(label_paths):
                raise Exception(f"Number of images does not match number of labels for slide {slide_id}")

            # otherwise go ahead and check files
            for i in tqdm.tqdm(range(len(img_paths))):
                tile = img_paths[i]
                filename = utils.get_filename(tile)
                anns_file = os.path.join(label_dir, f"{filename}.txt")

                if os.path.getsize(anns_file) == 0:
                    os.remove(anns_file)
                    os.remove(tile)

