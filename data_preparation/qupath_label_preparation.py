# Classes to read per-slide annotations exported from QuPath with `/qupath_scripts/export_detections.groovy`
# and export them into per-tile label files in YOLO format.

import os
import random
import tqdm
import data_utils
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


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
        detection_files = data_utils.list_files_of_a_type(
            self.root_label_dir, ".txt")

        for i in tqdm.tqdm(range(len(detection_files))):
            detection_file = detection_files[i]

            print(f"\nProcessing {detection_file}...")

            # check this is not a filtered annotation file as denoted by our naming convention
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] == "filtered":
                continue

            slide_id = split_filename[0]

            with open(detection_file, 'r') as detections:
                output_path = os.path.join(
                    self.out_dir, f"{slide_id}_detections_filtered.txt")

                with open(output_path, 'w') as filtered_detections:
                    for line in detections:
                        object_label = line.split(
                            ':')[0].strip().strip(']').strip('[')
                        if object_label in self.class_to_idx.keys():
                            filtered_detections.write(line)

    def _visualize_samples(self, vis_samples, vis_dir, length_threshold=None, area_thresholds=None):
        """
        Visualize sample objects that are:
        - For area threshold: within ±0.05 of their class-specific area threshold
        - For length threshold: within ±5 pixels of the length threshold
        Saves each sample as a separate image.

        Args:
            vis_samples: Dictionary containing kept/deleted samples
            vis_dir: Directory to save visualizations
            length_threshold: Length threshold in pixels
            area_thresholds: List of area thresholds per class
        """

        # Filter samples near their respective thresholds
        boundary_samples = {
            'kept': [],
            'deleted': []
        }

        for category in ['kept', 'deleted']:
            for sample in vis_samples[category]:
                should_visualize = False

                # Check if sample is near threshold
                if 'area_ratio' in sample and area_thresholds is not None:
                    # For area threshold
                    if abs(sample['area_ratio'] - sample['threshold']) <= 0.05:
                        should_visualize = True

                if 'min_length' in sample and length_threshold is not None:
                    # For length threshold
                    if abs(sample['min_length'] - length_threshold) <= 5 and len(boundary_samples[category]) < 30:
                        should_visualize = True

                if should_visualize:
                    boundary_samples[category].append(sample)

        # Count total samples
        total_samples = len(
            boundary_samples['kept']) + len(boundary_samples['deleted'])
        if total_samples == 0:
            print("No samples found near the threshold boundaries.")
            return

        # Create subdirectories for kept and deleted samples
        for category in ['kept', 'deleted']:
            category_dir = os.path.join(vis_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

        # Save summary of thresholds
        with open(os.path.join(vis_dir, 'thresholds.txt'), 'w') as f:
            if area_thresholds is not None:
                f.write(f"Class-specific area thresholds:\n")
                for idx, threshold in enumerate(area_thresholds):
                    f.write(f"Class {idx}: {threshold:.3f}\n")
            if length_threshold is not None:
                f.write(f"\nLength threshold: {length_threshold} pixels\n")
            f.write(f"\nTotal boundary samples: {total_samples}")

        # Plot and save each sample individually
        for category in ['kept', 'deleted']:
            for i, sample in enumerate(sorted(boundary_samples[category],
                                              key=lambda x: (x['class_idx'],
                                                             x.get('area_ratio', 0) if 'area_ratio' in x
                                                             else x.get('min_length', 0)))):
                # Create figure for this sample
                fig, ax = plt.subplots(figsize=(8, 8))

                # Load and display image
                img = Image.open(sample['img_path'])
                ax.imshow(img)

                # Draw bounding box
                bbox = sample['bbox']
                w, h = img.size
                rect = patches.Rectangle(
                    (bbox[0]*w - bbox[2]*w/2, bbox[1]*h - bbox[3]*h/2),
                    bbox[2]*w, bbox[3]*h,
                    linewidth=2,
                    edgecolor='g' if category == 'kept' else 'r',
                    facecolor='none'
                )
                ax.add_patch(rect)

                # Add title with detailed information
                title_parts = [
                    f"{sample['class']} (idx: {sample['class_idx']})",
                    f"{category.capitalize()}"
                ]

                if 'area_ratio' in sample:
                    title_parts.extend([
                        f"Area Ratio: {sample['area_ratio']:.3f}",
                        f"Area Threshold: {sample['threshold']:.3f}"
                    ])

                if 'min_length' in sample:
                    title_parts.extend([
                        f"Min Length: {sample['min_length']:.1f}px",
                        f"Length Threshold: {length_threshold}px"
                    ])

                ax.set_title('\n'.join(title_parts), pad=15)
                ax.axis('off')

                # Save figure
                threshold_type = 'area' if 'area_ratio' in sample else 'length'
                threshold_value = (f"{sample['area_ratio']:.3f}" if 'area_ratio' in sample
                                   else f"{sample['min_length']:.1f}")

                filename = f"class{sample['class_idx']}_{category}_{i:03d}_{threshold_type}{threshold_value}.png"
                plt.savefig(os.path.join(vis_dir, category, filename),
                            bbox_inches='tight',
                            dpi=300)
                plt.close(fig)

    def filter_files_with_no_labels(self, max=10000, inplace=False):
        """
        Filters out all empty per-tile annotation files and the corresponding tile image files.

        Args:
            max (int): Maximum number of empty files to remove (default: 10000)
            inplace (bool): If True, removes files from original directories instead of copying to new ones (default: False)
        """
        count = 0
        img_dirs = [f for f in os.listdir(self.root_img_dir) if os.path.isdir(os.path.join(
            self.root_img_dir, f))] if os.path.isdir(self.root_img_dir) else [self.root_img_dir]

        for slide_id in img_dirs:

            print(f"\nProcessing slide {slide_id}...")
            img_dir = os.path.join(self.root_img_dir, slide_id)
            label_dir = os.path.join(self.out_dir, slide_id)
            print(label_dir)

            if not inplace:
                # Create output directories if not operating inplace
                label_dir_kept = os.path.join(self.out_dir, slide_id+'kept')
                img_dir_kept = os.path.join(self.root_img_dir, slide_id+'kept')
                if not os.path.exists(label_dir_kept):
                    os.makedirs(label_dir_kept)
                if not os.path.exists(img_dir_kept):
                    os.makedirs(img_dir_kept)

            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")
            label_paths = data_utils.list_files_of_a_type(label_dir, ".txt")
            factor = 2 if self.with_segmentation else 1

            # check that every image has a label file
            if not self.preprocessed_labels and len(img_paths) != len(label_paths) * factor:
                print(f"in image dir: {len(img_paths)}")
                print(f"in label dir: {len(label_paths)}")
                raise Exception(
                    f"Number of images does not match number of labels for slide {slide_id}")

            # Process files
            for i in tqdm.tqdm(range(len(img_paths))):
                tile = img_paths[i]
                filename = data_utils.get_filename(tile)
                anns_file = os.path.join(label_dir, f"{filename}.txt")

                if not os.path.exists(anns_file):
                    continue

                if os.path.getsize(anns_file) != 0 or count >= max:
                    if not inplace:
                        shutil.copy2(tile, img_dir_kept)
                        if not self.with_segmentation:
                            shutil.copy2(anns_file, label_dir_kept)
                else:
                    if inplace:
                        os.remove(tile)
                        os.remove(anns_file)
                    count += 1

    def add_empty_tiles(self, empty_tiles_required: dict[str, int]):
        """
        Adds empty tiles to the training set to balance the number of objects per class.

        Args:
            empty_tiles_required (dict[str, int]): Dictionary mapping slide IDs to the number of empty tiles to add.
        """
        for slide_id, num_empty_tiles in empty_tiles_required.items():
            img_dir = os.path.join(self.root_img_dir, slide_id)
            label_dir = os.path.join(self.out_dir, 'train', slide_id)
            img_dir_kept = os.path.join(self.root_img_dir, slide_id+'kept')
            label_dir_kept = os.path.join(self.out_dir, slide_id)
            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")
            # sample empty_tiles_required number of empty tiles from img_dir, that are not already in img_dir_kept, and copy them to img_dir_kept. copy the corresponding label files to label_dir_kept.
            i = 0
            while i < num_empty_tiles:
                empty_tile = random.choice(img_paths)
                if empty_tile not in img_dir_kept:
                    shutil.copy2(empty_tile, img_dir_kept)
                    shutil.copy2(os.path.join(
                        label_dir, f"{data_utils.get_filename(empty_tile)}.txt"), label_dir_kept)
                    i += 1

    def _validate_thresholds(self, area_thresholds=None, length_threshold=None):
        """
        Validates area and length thresholds.

        Args:
            area_thresholds (list[float], optional): List of area ratio thresholds for each class
            length_threshold (int, optional): Minimum length threshold in pixels
            
        Returns:
            bool: True if validation passes, False otherwise
            
        Raises:
            ValueError: If area_thresholds is provided but doesn't have exactly 4 values
        """
        if area_thresholds is not None and len(area_thresholds) != 4:
            raise ValueError("area_thresholds must be a list of 4 values for class indices 0-3")

        return True

    def _get_intersection_coords(self, label, tile_bounds):
        """
        Calculate intersection coordinates between a label and a tile.

        Args:
            label: Label object containing bounding box information
            tile_bounds: Tuple of (x0, y0, width, height) for the tile
            
        Returns:
            tuple: Intersection coordinates (x_start, x_end, y_start, y_end)
        """
        tile_x0, tile_y0, tile_w, tile_h = tile_bounds

        x_intersection_start = max(label.x0, tile_x0)
        x_intersection_end = min(label.x0 + label.w, tile_x0 + tile_w)
        y_intersection_start = max(label.y0, tile_y0)
        y_intersection_end = min(label.y0 + label.h, tile_y0 + tile_h)

        return (x_intersection_start, x_intersection_end, y_intersection_start, y_intersection_end)

    def _has_intersection(self, intersection_coords):
        """
        Check if there is an actual intersection between two rectangles.

        Args:
            intersection_coords: Tuple of (x_start, x_end, y_start, y_end)
            
        Returns:
            bool: True if there is an intersection, False otherwise
        """
        x_start, x_end, y_start, y_end = intersection_coords
        return x_start < x_end and y_start < y_end

    def _calculate_areas(self, intersection_coords, label):
        """
        Calculate intersection area and box area.

        Args:
            intersection_coords: Tuple of (x_start, x_end, y_start, y_end)
            label: Label object containing bounding box information
            
        Returns:
            tuple: (intersection_area, box_area, intersection_dimensions)
        """
        x_start, x_end, y_start, y_end = intersection_coords

        intersection_width = x_end - x_start
        intersection_height = y_end - y_start
        intersection_area = intersection_width * intersection_height
        box_area = label.w * label.h

        return (intersection_area, box_area, (intersection_width, intersection_height))

    def _calculate_normalized_coords(self, intersection_coords, tile_bounds):
        """
        Calculate normalized coordinates for YOLO format.

        Args:
            intersection_coords: Tuple of (x_start, x_end, y_start, y_end)
            tile_bounds: Tuple of (x0, y0, width, height) for the tile
            
        Returns:
            tuple: (x_centre, y_centre, norm_width, norm_height)
        """
        x_start, x_end, y_start, y_end = intersection_coords
        tile_x0, tile_y0, tile_w, tile_h = tile_bounds

        x_centre = ((x_start + x_end) / 2 - tile_x0) / tile_w
        y_centre = ((y_start + y_end) / 2 - tile_y0) / tile_h
        norm_width = (x_end - x_start) / tile_w
        norm_height = (y_end - y_start) / tile_h

        return (x_centre, y_centre, norm_width, norm_height)

    def _should_filter_object(self, intersection_area, box_area, dimensions, area_thresholds=None, length_threshold=None, class_idx=None):
        """
        Determine if an object should be filtered based on thresholds.

        Args:
            intersection_area: Area of intersection
            box_area: Total area of bounding box
            dimensions: Tuple of (width, height) of the intersection
            area_thresholds: List of area thresholds per class
            length_threshold: Minimum length threshold
            class_idx: Class index of the object
            
        Returns:
            bool: True if object should be filtered out, False if it should be kept
        """
        # If the intersection is the entire box, always keep
        if intersection_area == box_area:
            return False
            
        # Check area threshold first if provided
        if area_thresholds is not None and class_idx is not None:
            area_ratio = intersection_area / box_area
            if area_ratio < area_thresholds[class_idx]:
                return True
                
        # Then check length threshold if provided
        if length_threshold is not None:
            width, height = dimensions
            return width < length_threshold or height < length_threshold
                
        # Default to keeping the object if no threshold criteria matched
        return False

    def _parse_tile_info(self, filename):
        """
        Parse tile filename to extract tile bounds.

        Args:
            filename: Tile filename
            
        Returns:
            tuple: (x0, y0, width, height)
        """
        tile_info = filename.split(',')
        tile_x0 = int(tile_info[1][2:])
        tile_y0 = int(tile_info[2][2:])
        tile_w = int(tile_info[3][2:])
        tile_h = int(tile_info[4][2:-1])

        return (tile_x0, tile_y0, tile_w, tile_h)

    def _write_statistics(self, statistics, threshold_type, threshold_value, file_path=None):
        """
        Write filtering statistics to a file.

        Args:
            statistics: Dictionary containing statistics
            threshold_type: Type of threshold used
            threshold_value: Value of threshold
            file_path: Path to output file (optional)
        """
        if file_path is None:
            file_path = os.path.join(self.out_dir, f'separation_statistics_{threshold_type}_{threshold_value}.csv')
            
        with open(file_path, 'w') as stats_f:
            # Write header
            stats_f.write("slide_id,class_idx,kept,deleted\n")
            # Write data
            for slide_id, class_stats in statistics.items():
                for class_idx, counts in class_stats.items():
                    stats_f.write(f"{slide_id},{class_idx},{counts['kept']},{counts['deleted']}\n")

    def separate_labels_by_tile(self, area_thresholds=None, length_threshold=None):
        """
        Reads all filtered per-slide annotation files in `self.out_dir` and writes out per-tile annotation files
        for each image tile in `self.root_img_dir`.

        Can filter objects based on either:
        - Length threshold: minimum length of visible portion in pixels
        - Area thresholds: minimum ratio of visible area to original area, specified per class

        Args:
            area_thresholds (list[float], optional): List of area ratio thresholds for each class index [0,1,2,3]
            length_threshold (int, optional): Minimum length threshold in pixels

        If both thresholds are provided, area_thresholds takes precedence.
        If neither is provided, no filtering is applied.

        Output label files are grouped into directories in `self.out_dir` by slide.

        Returns:
            dict: Statistics dictionary
        """
        self._validate_thresholds(area_thresholds, length_threshold)

        statistics = {}
        detection_files = data_utils.list_files_of_a_type(self.out_dir, ".txt")

        for detection_file in tqdm.tqdm(detection_files):
            # Check this is a filtered annotation file as denoted by our naming convention
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] != "filtered.txt":
                continue

            # Get the ID of the slide that this annotation file is for
            slide_id = split_filename[0]
            if slide_id[0] == '~':
                continue
                
            # Initialize statistics for this slide
            statistics[slide_id] = {class_idx: {'kept': 0, 'deleted': 0} 
                                    for class_name, class_idx in self.class_to_idx.items()}

            img_dir = os.path.join(self.root_img_dir, slide_id)
            out_label_dir = os.path.join(self.out_dir, slide_id)
            if not os.path.exists(out_label_dir):
                os.makedirs(out_label_dir)

            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")

            for tile in img_paths:
                filename = data_utils.get_filename(tile)
                tile_bounds = self._parse_tile_info(filename)
                
                with open(detection_file, 'r') as all_anns:
                    all_anns.readline()  # Skip header line
                    out_label_file = os.path.join(out_label_dir, f"{filename}.txt")

                    with open(out_label_file, 'w') as label_f:
                        for obj in all_anns:
                            # Process the line into a Label object
                            label = Label(obj, self.class_to_idx)
                            if label.class_idx == -1:  # Label is not in a class of interest, skip
                                continue

                            # Get intersection between label and tile
                            intersection_coords = self._get_intersection_coords(label, tile_bounds)
                            
                            # Check if there is any intersection
                            if not self._has_intersection(intersection_coords):
                                continue
                                
                            # Calculate areas and dimensions
                            intersection_area, box_area, dimensions = self._calculate_areas(
                                intersection_coords, label)
                                
                            # Determine whether to filter this object
                            should_delete = self._should_filter_object(
                                intersection_area, box_area, dimensions, 
                                area_thresholds, length_threshold, label.class_idx)

                            # Update statistics
                            if should_delete:
                                statistics[slide_id][label.class_idx]['deleted'] += 1
                                continue
                            else:
                                statistics[slide_id][label.class_idx]['kept'] += 1

                            # Calculate normalized coordinates for YOLO format
                            x_centre, y_centre, norm_width, norm_height = self._calculate_normalized_coords(
                                intersection_coords, tile_bounds)

                            # Write YOLO format label
                            label_yolo = f"{label.class_idx} {x_centre} {y_centre} {norm_width} {norm_height}\n"
                            label_f.write(label_yolo)

        # Save statistics
        threshold_type = "area" if area_thresholds is not None else "length" if length_threshold is not None else "none"
        threshold_value = str(area_thresholds) if area_thresholds is not None else str(
            length_threshold) if length_threshold is not None else "none"

        self._write_statistics(statistics, threshold_type, threshold_value)

        return statistics

    def generate_cut_log_per_tile(self, length_threshold=None, area_thresholds=None, tile_number=None):
        """
        Reads filtered per-slide annotation files and generates visualizations for objects near the thresholds.

        Args:
            length_threshold (int, optional): Minimum length threshold in pixels
            area_thresholds (list[float], optional): List of thresholds for each class index [0,1,2,3]
            tile_number (int, optional): Specific tile number to process
            
        Returns:
            dict: Statistics dictionary
        """
        # Validate input parameters
        if not length_threshold and not area_thresholds:
            raise ValueError("At least one of length_threshold or area_thresholds must be provided")
            
        self._validate_thresholds(area_thresholds)

        # Create reverse mapping from class_idx to class_name
        idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        # Initialize statistics
        statistics = {
            'length': {name: {'kept': 0, 'deleted': 0} for name in self.class_to_idx.keys()},
            'area': {name: {'kept': 0, 'deleted': 0} for name in self.class_to_idx.keys()}
        }

        # For visualization samples
        vis_samples = {
            'length': {'kept': [], 'deleted': []},
            'area': {'kept': [], 'deleted': []}
        }

        # Create visualization directory
        vis_dir = os.path.join(self.out_dir, str(tile_number) if tile_number else "all", 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        detection_files = data_utils.list_files_of_a_type(self.out_dir, ".txt")

        for detection_file in tqdm.tqdm(detection_files):
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] != "filtered.txt":
                continue
                
            slide_id = split_filename[0]
            # Skip if tile_number is specified and doesn't match
            if tile_number is not None and slide_id != str(tile_number):
                continue
                
            img_dir = os.path.join(self.root_img_dir, slide_id+'kept')
            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")

            for tile in img_paths:
                filename = data_utils.get_filename(tile)
                tile_bounds = self._parse_tile_info(filename)

                with open(detection_file, 'r') as all_anns:
                    all_anns.readline()  # Skip header

                    for obj in all_anns:
                        label = Label(obj, self.class_to_idx)
                        if label.class_idx == -1:  # Not a class of interest
                            continue

                        # Calculate intersection
                        intersection_coords = self._get_intersection_coords(label, tile_bounds)
                        
                        # Check if there's an intersection
                        if not self._has_intersection(intersection_coords):
                            continue
                            
                        # Calculate areas and dimensions
                        intersection_area, box_area, dimensions = self._calculate_areas(
                            intersection_coords, label)
                        visible_width, visible_height = dimensions
                            
                        # Calculate normalized coordinates for visualization
                        norm_coords = self._calculate_normalized_coords(intersection_coords, tile_bounds)
                        x_centre, y_centre, norm_width, norm_height = norm_coords

                        # Process length threshold
                        if length_threshold:
                            min_length = min(visible_width, visible_height)
                            # Use the helper function to determine if object should be filtered based on length
                            length_filter_only = self._should_filter_object(
                                intersection_area, box_area, dimensions, 
                                length_threshold=length_threshold)
                            category = 'deleted' if length_filter_only else 'kept'
                            
                            # Only collect samples if they're within 3 pixels of threshold
                            if abs(min_length - length_threshold) <= 3:
                                vis_samples['length'][category].append({
                                    'img_path': tile,
                                    'bbox': [x_centre, y_centre, norm_width, norm_height],
                                    'class': idx_to_class[label.class_idx],
                                    'class_idx': label.class_idx,
                                    'min_length': min_length
                                })
                            statistics['length'][idx_to_class[label.class_idx]][category] += 1

                        # Process area threshold if object was kept by length criteria or no length threshold exists
                        if (not length_threshold or category == 'kept') and area_thresholds:
                            # Use the helper function to determine if object should be filtered based on area
                            area_filter_only = self._should_filter_object(
                                intersection_area, box_area, dimensions,
                                area_thresholds=area_thresholds, class_idx=label.class_idx)
                            category = 'deleted' if area_filter_only else 'kept'
                            
                            # Only collect samples if they're deleted and within 0.05 of threshold
                            if category == 'kept':
                                continue
                                
                            area_ratio = intersection_area / box_area
                            threshold = area_thresholds[label.class_idx]
                            
                            if abs(area_ratio - threshold) <= 0.05:
                                vis_samples['area'][category].append({
                                    'img_path': tile,
                                    'bbox': [x_centre, y_centre, norm_width, norm_height],
                                    'class': idx_to_class[label.class_idx],
                                    'class_idx': label.class_idx,
                                    'area_ratio': area_ratio,
                                    'threshold': threshold
                                })
                            statistics['area'][idx_to_class[label.class_idx]][category] += 1

        # Save statistics for each threshold type
        for thresh_type in statistics:
            if (thresh_type == 'length' and length_threshold) or (thresh_type == 'area' and area_thresholds):
                stats_file = os.path.join(self.out_dir, f'{thresh_type}_threshold_statistics.txt')
                with open(stats_file, 'w') as stats_f:
                    stats_f.write(f"{thresh_type.capitalize()} threshold statistics:\n\n")
                    for class_name, class_stats in statistics[thresh_type].items():
                        total = class_stats['kept'] + class_stats['deleted']
                        if total > 0:
                            kept_percent = (class_stats['kept'] / total) * 100
                            stats_f.write(f"{class_name}:\n")
                            stats_f.write(f"  Kept: {class_stats['kept']} ({kept_percent:.1f}%)\n")
                            stats_f.write(f"  Deleted: {class_stats['deleted']} ({100-kept_percent:.1f}%)\n")
                            stats_f.write(f"  Total: {total}\n\n")

        # Visualize samples
        if length_threshold:
            self._visualize_samples(vis_samples['length'], os.path.join(vis_dir, 'length'),
                                    length_threshold=length_threshold)
        if area_thresholds:
            self._visualize_samples(vis_samples['area'], os.path.join(vis_dir, 'area'),
                                    area_thresholds=area_thresholds)

        return statistics
