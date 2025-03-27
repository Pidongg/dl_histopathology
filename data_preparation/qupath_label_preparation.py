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

                

    def generate_cut_log_per_tile(self, length_threshold: int = None, area_thresholds: list[float] = None, tile_number: int = None):
        """
        Reads filtered per-slide annotation files and generates visualizations for objects near the thresholds.
        
        Args:
            length_threshold (int, optional): Minimum length threshold in pixels
            area_thresholds (list[float], optional): List of thresholds for each class index [0,1,2,3]
            tile_number (int, optional): Specific tile number to process
        """
        if not length_threshold and not area_thresholds:
            raise ValueError("At least one of length_threshold or area_thresholds must be provided")
        
        if area_thresholds and len(area_thresholds) != 4:
            raise ValueError("area_thresholds must be a list of 4 values for class indices 0-3")

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
        vis_dir = os.path.join(self.out_dir, str(tile_number), 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        detection_files = data_utils.list_files_of_a_type(self.out_dir, ".txt")

        for detection_file in tqdm.tqdm(detection_files):
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] != "filtered.txt" or (tile_number is not None and split_filename[0] != str(tile_number)):
                continue

            slide_id = split_filename[0]
            if slide_id in ['747309','747316','747352'] or slide_id[0] == '~':
                continue
            img_dir = os.path.join(self.root_img_dir, slide_id+'kept')
            img_paths = data_utils.list_files_of_a_type(img_dir, ".png")

            for tile in img_paths:
                filename = data_utils.get_filename(tile)
                
                # Process filename to get tile bounds
                tile_info = filename.split(',')
                tile_x0 = int(tile_info[1][2:])
                tile_y0 = int(tile_info[2][2:])
                tile_w = int(tile_info[3][2:])
                tile_h = int(tile_info[4][2:-1])

                with open(detection_file, 'r') as all_anns:
                    all_anns.readline()  # skip header

                    for obj in all_anns:
                        label = Label(obj, self.class_to_idx)
                        if label.class_idx == -1:
                            continue

                        # Calculate intersection
                        x_intersection_start = max(label.x0, tile_x0)
                        x_intersection_end = min(label.x0 + label.w, tile_x0 + tile_w)
                        y_intersection_start = max(label.y0, tile_y0)
                        y_intersection_end = min(label.y0 + label.h, tile_y0 + tile_h)

                        if x_intersection_start < x_intersection_end and y_intersection_start < y_intersection_end:
                            intersection_area = (x_intersection_end - x_intersection_start) * (y_intersection_end - y_intersection_start)
                            box_area = label.w * label.h
                            
                            # Calculate normalized coordinates
                            x_centre = ((x_intersection_start + x_intersection_end) / 2 - tile_x0) / tile_w
                            y_centre = ((y_intersection_start + y_intersection_end) / 2 - tile_y0) / tile_h
                            norm_width = (x_intersection_end - x_intersection_start) / tile_w
                            norm_height = (y_intersection_end - y_intersection_start) / tile_h
                            
                            # Process length threshold
                            if length_threshold:
                                visible_width = x_intersection_end - x_intersection_start
                                visible_height = y_intersection_end - y_intersection_start
                                min_length = min(visible_width, visible_height)
                                category = 'deleted' if (intersection_area != box_area and min_length < length_threshold) else 'kept'
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

                            # Process area threshold
                            if category == 'kept' and area_thresholds:
                                area_ratio = intersection_area / box_area
                                threshold = area_thresholds[label.class_idx]
                                category = 'deleted' if (intersection_area != box_area and area_ratio < threshold) else 'kept'
                                # Only collect samples if they're within 0.05 of threshold
                                if category == 'kept':
                                    continue
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
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        
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
                    if abs(sample['min_length'] - length_threshold) <= 5 and len(boundary_samples[category])<30:
                        should_visualize = True
                
                if should_visualize:
                    boundary_samples[category].append(sample)
        
        # Count total samples
        total_samples = len(boundary_samples['kept']) + len(boundary_samples['deleted'])
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
        
        print(f"Saved {total_samples} visualization samples to {vis_dir}")
    

    def separate_labels_by_tile(self, area_thresholds: list[float] = None, length_threshold: int = None):
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
        """
        if area_thresholds is not None and len(area_thresholds) != 4:
            raise ValueError("area_thresholds must be a list of 4 values for class indices 0-3")

        statistics = {}
        detection_files = data_utils.list_files_of_a_type(self.out_dir, ".txt")

        for i in tqdm.tqdm(range(len(detection_files))):
            detection_file = detection_files[i]

            # check this is a filtered annotation file as denoted by our naming convention
            split_filename = data_utils.get_filename(detection_file).split('_')
            if split_filename[-1] != "filtered.txt":
                continue

            # get the id of the slide that this annotation file is for
            slide_id = split_filename[0]
            if slide_id[0] == '~':
                continue 
            statistics[slide_id] = {}
            for class_name in self.class_to_idx.keys():
                statistics[slide_id][self.class_to_idx[class_name]] = {'kept': 0, 'deleted': 0}

            img_dir = os.path.join(self.root_img_dir, slide_id) # TODO: remove kept
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
                                
                                # Determine whether to keep the object based on the provided threshold type
                                should_delete = False
                                
                                if intersection_area != box_area and area_thresholds is not None:
                                    # Use area ratio threshold
                                    area_ratio = intersection_area / box_area
                                    should_delete = area_ratio < area_thresholds[label.class_idx]
                                if intersection_area != box_area and should_delete is False and length_threshold is not None:
                                    should_delete = ((x_intersection_end - x_intersection_start) < length_threshold or (y_intersection_end - y_intersection_start) < length_threshold)
                                
                                if should_delete:
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

        # Save statistics
        threshold_type = "area" if area_thresholds is not None else "length" if length_threshold is not None else "none"
        threshold_value = str(area_thresholds) if area_thresholds is not None else str(length_threshold) if length_threshold is not None else "none"
        
        stats_file = os.path.join(self.out_dir, f'separation_statistics_{threshold_type}_{threshold_value}.csv')
        with open(stats_file, 'w') as stats_f:
            # Write header
            stats_f.write("slide_id,class_idx,kept,deleted\n")
            # Write data
            for slide_id, class_stats in statistics.items():
                for class_idx, counts in class_stats.items():
                    stats_f.write(f"{slide_id},{class_idx},{counts['kept']},{counts['deleted']}\n")

        return statistics

    def filter_files_with_no_labels(self, max=10000, inplace=False):
        """
        Filters out all empty per-tile annotation files and the corresponding tile image files.

        Args:
            max (int): Maximum number of empty files to remove (default: 10000)
            inplace (bool): If True, removes files from original directories instead of copying to new ones (default: False)
        """
        count = 0
        img_dirs = [f for f in os.listdir(self.root_img_dir) if os.path.isdir(os.path.join(self.root_img_dir, f))] if os.path.isdir(self.root_img_dir) else [self.root_img_dir]
        
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
                raise Exception(f"Number of images does not match number of labels for slide {slide_id}")

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
                    shutil.copy2(os.path.join(label_dir, f"{data_utils.get_filename(empty_tile)}.txt"), label_dir_kept)
                    i += 1
