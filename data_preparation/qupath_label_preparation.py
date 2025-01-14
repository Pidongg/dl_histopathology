# Classes to read per-slide annotations exported from QuPath with `/qupath_scripts/export_detections.groovy`
# and export them into per-tile label files in YOLO format.

import os
import random
import tqdm
import data_utils

import shutil



class Label:
    """
    A class to encapsulate the definition of a single object annotation.

    Args:
        label_text (str): A line of text from a QuPath-exported annotation file, corresponding to one object annotation.
        class_to_idx (dict[str, int]): A mapping from class names to indexes.

    Attributes:
        class_idx (int): Index of the box's class; -1 if invalid label
        x0 (int): x-coordinate of the box's upper left corner in pixels
        y0 (int): y-coordinate of the box's upper left corner in pixels
        w (int): Width of the box in pixels
        h (int): Height of the box in pixels
    """

    def __init__(self, label_text: str, class_to_idx: dict[str, int]):
        class_name, roi = label_text.strip().split(':')
        class_name = class_name.strip('[] ')

        if class_name in class_to_idx:
            # if the class name is included in the provided dictionary, then extract information.
            self.class_idx = class_to_idx[class_name]
            obj_info = roi[roi.find('('):].strip('()').split(',')
            self.x0 = int(obj_info[0])
            self.y0 = int(obj_info[1])
            self.w = int(obj_info[2])
            self.h = int(obj_info[3])
        else:
            # if the class name is not in the provided dictionary, set default values to indicate an invalid label.
            self.x0 = 0
            self.y0 = 0
            self.w = 0
            self.h = 0
            self.class_idx = -1


class LabelPreparer:
    """
    A class to process the per-slide annotation files output by `/qupath_scripts/export_detections.groovy` and create
        individual label files for each tile of the corresponding slide image.

    Note that all methods expect the following naming conventions: per-slide annotations directly exported from QuPath
        as "{slide_id}_detections.txt", and filtered per-slide annotations as "{slide_id}_filtered_detections.txt".

    Args:
        root_label_dir (path): Path to the (flat) directory holding the annotation files to be processed.
        root_img_dir (path): Path to the directory holding tiles of the corresponding slides, organised by slide.
        out_dir (path): Path to the directory to output label files to.
        class_to_idx (dict[str, int]): Mapping from class names to indices.

    Attributes:
        root_label_dir (path), root_img_dir (path), out_dir (path), class_to_idx (dict[str, int])

    Methods:
        remove_unlabelled_objects: Write out per-slide annotation files that contain no unlabelled detections.
        separate_labels_by_tile: Writes per-tile annotation files for each tile in the image directory.
        delete_files_with_no_labels: Deletes all tiles with empty annotation files along with their annotation files.
    """
    def __init__(self, root_label_dir: os.path, root_img_dir: os.path, out_dir: os.path, class_to_idx: dict[str, int], with_segmentation: bool, preprocessed_labels=False):
        self.root_label_dir = root_label_dir
        self.root_img_dir = root_img_dir
        self.out_dir = out_dir
        self.with_segmentation = with_segmentation

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.class_to_idx = class_to_idx
        self.preprocessed_labels = preprocessed_labels

    def remove_unlabelled_objects(self):
        """
        Reads all per-slide annotation files in `self.root_label_dir` and writes out a new per-slide annotation file
            to `self.out_dir`, containing only labelled object annotations (i.e. class != "unlabelled" or "others").
        """
        detection_files = data_utils.list_files_of_a_type(self.root_label_dir, ".txt")

        for i in tqdm.tqdm(range(len(detection_files))):
            detection_file = detection_files[i]

            print(f"\nProcessing {detection_file}...")

            # check this is not a filtered annotation file as denoted by our naming convention
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] == "filtered":
                continue

            slide_id = split_filename[0]

            with open(detection_file, 'r') as detections:
                output_path = os.path.join(self.out_dir, f"{slide_id}_detections_filtered.txt")

                with open(output_path, 'w') as filtered_detections:
                    for line in detections:
                        object_label = line.split(':')[0].strip().strip(']').strip('[')
                        if object_label in self.class_to_idx.keys():
                            filtered_detections.write(line)

    def separate_tiles_with_cut_log(self):
        # list subdirectories in self.out_dir
        slide_dirs = [f for f in os.listdir(self.out_dir) if os.path.isdir(os.path.join(self.out_dir, f))]
        for slide_dir in slide_dirs:
            detection_files = data_utils.list_files_of_a_type(os.path.join(self.out_dir, slide_dir), ".txt")
            out_label_dir = os.path.join(self.out_dir, 'cut_off_images', slide_dir)
            if not os.path.exists(out_label_dir):
                os.makedirs(out_label_dir)
            for i in tqdm.tqdm(range(len(detection_files))):
                detection_file = detection_files[i]
                if open(detection_file, 'r').readline() == "":
                    continue
                # check this is a filtered annotation file as denoted by our naming convention
                split_filename = data_utils.get_filename(detection_file).split('_')
                img_dir = os.path.join(self.root_img_dir, slide_dir)
                shutil.copy2(os.path.join(img_dir, split_filename[0]+'.png'), out_label_dir)

                

    def generate_cut_log_per_tile_with_length_threshold(self, length_threshold: int, tile_number: int):
        """
        Reads all filtered per-slide annotation files in `self.out_dir` and writes out per-tile annotation files
        for each image tile in `self.root_img_dir`.

        Output label files are grouped into directories in `self.out_dir` by slide.
        """
        statistics = {}
        for class_name in self.class_to_idx.keys():
            statistics[class_name] = {'kept': 0, 'deleted': 0}
        detection_files = data_utils.list_files_of_a_type(self.out_dir, ".txt")

        for i in tqdm.tqdm(range(len(detection_files))):
            detection_file = detection_files[i]

            # check this is a filtered annotation file as denoted by our naming convention
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] != "filtered" or split_filename[0] != str(tile_number):
                continue

            # get the id of the slide that this annotation file is for
            slide_id = split_filename[0]

            img_dir = os.path.join(self.root_img_dir, slide_id)
            out_label_dir = os.path.join(self.out_dir, slide_id, 'kept_with_length_threshold', str(length_threshold))
            deleted_label_dir = os.path.join(self.out_dir, slide_id, 'deleted_with_length_threshold', str(length_threshold))
            if not os.path.exists(out_label_dir):
                os.makedirs(out_label_dir)
            if not os.path.exists(deleted_label_dir):
                os.makedirs(deleted_label_dir)

            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")

            for tile in img_paths:
                filename = data_utils.get_filename(tile)

                # process filename to get tile bound info
                tile_info = filename.split(',')
                tile_x0 = int(tile_info[1][2:])
                tile_y0 = int(tile_info[2][2:])
                tile_w = int(tile_info[3][2:])
                tile_h = int(tile_info[4][2:-1])

                with open(detection_file, 'r') as all_anns:
                    all_anns.readline()  # skip first line

                    cut_log_file = os.path.join(out_label_dir, f"{filename}_cut_log.txt")
                    cut_log_file_deleted = os.path.join(deleted_label_dir, f"{filename}_cut_log.txt")

                    with open(cut_log_file, 'w') as cut_log_f:
                        with open(cut_log_file_deleted, 'w') as cut_log_f_deleted:
                            for obj in all_anns:
                                label = Label(obj, self.class_to_idx)  # process the line into a Label object

                                if label.class_idx == -1:  # label is not in a class of interest, so skip
                                    continue
                                x_intersection_start = max(label.x0, tile_x0)
                                x_intersection_end = min(label.x0 + label.w, tile_x0 + tile_w)
                                y_intersection_start = max(label.y0, tile_y0)
                                y_intersection_end = min(label.y0 + label.h, tile_y0 + tile_h)
                                
                                # Check if there is any intersection
                                if x_intersection_start < x_intersection_end and y_intersection_start < y_intersection_end:
                                    intersection_area = (x_intersection_end - x_intersection_start) * (y_intersection_end - y_intersection_start)
                                    box_area = label.w * label.h
                                    # log the cut in the detection file, and the length and width of the visible portion
                                    if intersection_area != box_area:
                                        x_centre = ((x_intersection_start + x_intersection_end) / 2 - tile_x0) / tile_w
                                        y_centre = ((y_intersection_start + y_intersection_end) / 2 - tile_y0) / tile_h
                                        # Use the visible portion for width and height
                                        norm_width = (x_intersection_end - x_intersection_start) / tile_w
                                        norm_height = (y_intersection_end - y_intersection_start) / tile_h
                                        if x_intersection_end - x_intersection_start < length_threshold or y_intersection_end - y_intersection_start < length_threshold :
                                            cut_log_f_deleted.write(f"{label.class_idx} {x_centre} {y_centre} {norm_width} {norm_height}\n")
                                            statistics[class_name]['deleted'] += 1
                                        else:
                                            cut_log_f.write(f"{label.class_idx} {x_centre} {y_centre} {norm_width} {norm_height}\n")
                                            statistics[class_name]['kept'] += 1
        with open(os.path.join(self.out_dir, 'statistics.txt'), 'w') as stats_f:
            for class_name, class_stats in statistics.items():
                stats_f.write(f"{class_name}: {class_stats['kept']} kept, {class_stats['deleted']} deleted\n")

    def separate_labels_by_tile(self, length_threshold: int):
        """
        Reads all filtered per-slide annotation files in `self.out_dir` and writes out per-tile annotation files
        for each image tile in `self.root_img_dir`.

        Output label files are grouped into directories in `self.out_dir` by slide.
        """
        statistics = {}
        detection_files = data_utils.list_files_of_a_type(self.out_dir, ".txt")

        for i in tqdm.tqdm(range(len(detection_files))):
            detection_file = detection_files[i]

            # check this is a filtered annotation file as denoted by our naming convention
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] != "filtered":
                continue

            # get the id of the slide that this annotation file is for
            slide_id = split_filename[0]
            statistics[slide_id] = {}
            for class_name in self.class_to_idx.keys():
                statistics[slide_id][self.class_to_idx[class_name]] = {'kept': 0, 'deleted': 0}

            img_dir = os.path.join(self.root_img_dir, slide_id)
            out_label_dir = os.path.join(self.out_dir, slide_id)
            if not os.path.exists(out_label_dir):
                os.makedirs(out_label_dir)

            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")

            for tile in img_paths:
                filename = data_utils.get_filename(tile)

                # process filename to get tile bound info
                tile_info = filename.split(',')
                tile_x0 = int(tile_info[1][2:])
                tile_y0 = int(tile_info[2][2:])
                tile_w = int(tile_info[3][2:])
                tile_h = int(tile_info[4][2:-1])

                with open(detection_file, 'r') as all_anns:
                    all_anns.readline()  # skip first line

                    out_label_file = os.path.join(out_label_dir, f"{filename}.txt")

                    with open(out_label_file, 'w') as label_f:
                        for obj in all_anns:
                            label = Label(obj, self.class_to_idx)  # process the line into a Label object

                            if label.class_idx == -1:  # label is not in a class of interest, so skip
                                continue
                            x_intersection_start = max(label.x0, tile_x0)
                            x_intersection_end = min(label.x0 + label.w, tile_x0 + tile_w)
                            y_intersection_start = max(label.y0, tile_y0)
                            y_intersection_end = min(label.y0 + label.h, tile_y0 + tile_h)
                            
                            # Check if there is any intersection
                            if x_intersection_start < x_intersection_end and y_intersection_start < y_intersection_end:
                                intersection_area = (x_intersection_end - x_intersection_start) * (y_intersection_end - y_intersection_start)
                                box_area = label.w * label.h
                                # log the cut in the detection file, and the length and width of the visible portion
                                if intersection_area != box_area and ((x_intersection_end - x_intersection_start) < length_threshold or (y_intersection_end - y_intersection_start) < length_threshold):
                                    statistics[slide_id][label.class_idx]['deleted'] += 1
                                    continue
                                else:
                                    statistics[slide_id][label.class_idx]['kept'] += 1  
                                # Calculate center point of the visible portion of the box
                                x_centre = ((x_intersection_start + x_intersection_end) / 2 - tile_x0) / tile_w
                                y_centre = ((y_intersection_start + y_intersection_end) / 2 - tile_y0) / tile_h

                                # Use the visible portion for width and height
                                norm_width = (x_intersection_end - x_intersection_start) / tile_w
                                norm_height = (y_intersection_end - y_intersection_start) / tile_h

                                label_yolo = f"{label.class_idx} {x_centre} {y_centre} {norm_width} {norm_height}\n"
                                label_f.write(label_yolo)
        with open(os.path.join(self.out_dir, 'statistics_2.csv'), 'w') as stats_f:
            for slide_id, class_stats in statistics.items():
                for class_name, class_stats in class_stats.items():
                    stats_f.write(f"{slide_id},{class_name},{class_stats['kept']},{class_stats['deleted']}\n")

    def filter_files_with_no_labels(self):
        """
        Filters out all empty per-tile annotation files and the corresponding tile image files.
        """
        # Get slide IDs.
        img_dirs = [f for f in os.listdir(self.root_img_dir) if os.path.isdir(os.path.join(self.root_img_dir, f))]
        print(img_dirs)
        for slide_id in img_dirs:
            print(f"\nProcessing slide {slide_id}...")
            img_dir = os.path.join(self.root_img_dir, slide_id)
            label_dir = os.path.join(self.out_dir, slide_id)
            print(label_dir)
            label_dir_kept = os.path.join(self.out_dir, slide_id+'kept')   
            if not os.path.exists(label_dir_kept):
                os.makedirs(label_dir_kept)
            img_dir_kept = os.path.join(self.root_img_dir, slide_id+'kept')
            if not os.path.exists(img_dir_kept):
                os.makedirs(img_dir_kept)   
            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")
            label_paths = data_utils.list_files_of_a_type(label_dir, ".txt")
            factor = 2 if self.with_segmentation else 1
            # check that every image has a label file.
            if not self.preprocessed_labels and len(img_paths) != len(label_paths) * factor:
                print(f"in image dir: {len(img_paths)}")
                print(f"in label dir: {len(label_paths)}")
                raise Exception(f"Number of images does not match number of labels for slide {slide_id}")

            # otherwise go ahead and check files
            for i in tqdm.tqdm(range(len(img_paths))):
                tile = img_paths[i]
                filename = data_utils.get_filename(tile)
                anns_file = os.path.join(label_dir, f"{filename}.txt")
                if self.preprocessed_labels:
                    if os.path.exists(anns_file):
                        shutil.copy2(tile, img_dir_kept)
                    continue 

                if os.path.getsize(anns_file) != 0:
                    shutil.copy2(tile, img_dir_kept)
                    shutil.copy2(anns_file, label_dir_kept)
    
    def add_empty_tiles(self, empty_tiles_required: dict[str, int]):
        for slide_id, num_empty_tiles in empty_tiles_required.items():
            img_dir = os.path.join(self.root_img_dir, slide_id)
            label_dir = os.path.join(self.out_dir, slide_id)
            img_dir_kept = os.path.join(self.root_img_dir, slide_id+'kept')
            label_dir_kept = os.path.join(self.out_dir, slide_id+'kept')
            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")
            label_paths = data_utils.list_files_of_a_type(label_dir, ".txt")
            # sample empty_tiles_required number of empty tiles from img_dir, that are not already in img_dir_kept, and copy them to img_dir_kept. copy the corresponding label files to label_dir_kept.
            i = 0
            while i < num_empty_tiles:
                empty_tile = random.choice(img_paths)
                if empty_tile not in img_dir_kept:
                    shutil.copy2(empty_tile, img_dir_kept)
                    shutil.copy2(os.path.join(label_dir, f"{data_utils.get_filename(empty_tile)}.txt"), label_dir_kept)
                    i += 1
