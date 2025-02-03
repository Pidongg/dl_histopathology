from dataset_preparation import TauPreparer

in_root_dir = "M:/Unused/TauCellDL"

empty_tiles_required = {
    "747297": 25,
    # "747309": 0, #val
    # "747316": 0, #val
    "747337": 54, #train
    "747350": 29, #train
    # "747352": 0, #val
    "747814": 60,
    "747818": 25,
    "771746": 38,
    "771791": 57
}
# remove empty tiles from validation later. 
# names:
#   0: TA
#   1: CB
#   2: NFT
#   3: tau_fragments

# data_preparer = TauPreparer(in_root_dir=in_root_dir, in_img_dir="images/validation", in_label_dir="labels/validation",
#                             prepared_root_dir=in_root_dir, prepared_img_dir="images/validation", prepared_label_dir="labels/validation")
# data_preparer.show_bboxes("", label_dir="M:/Unused/TauCellDL/labels/validation/747352", img_dir="M:/Unused/TauCellDL/images/validation/747352", ext="")
# # data_preparer.show_bboxes("", "M:/Unused/TauCellDL/images/747297", "M:/Unused/TauCellDL/labels_new")

# data_preparer.prepare_labels_for_yolo()
# data_preparer.train_test_val_split(train=0.8, test=0, valid=0.2)

tiles = {
    "train": {
        "bg": [747297, 747814, 747818],
        "cortical": [771746, 771791],
        "dn": [747350, 747337]},
    "validation": {
        "bg": [747309],
        "cortical": [747316],
        "dn": [747352]},
    "test": {
        "bg": [703488, 747821],
        "cortical": [747331, 771747],
        "dn": [747335, 771913]
    }
}

# data_preparer.separate_by_tiles_dict(tiles)

# data_preparer.show_bboxes("", "M:/Unused/TauCellDL/images/747309", "M:/Unused/TauCellDL/labels_new/747309")

# data_preparer_test = TauPreparer(in_root_dir=in_root_dir, in_img_dir="test_images_seg", in_label_dir="labels/test",
#                             prepared_root_dir=in_root_dir, prepared_img_dir="test_images_seg", prepared_label_dir="labels/test",
#                             with_segmentation=True, preprocessed_labels=True)
# data_preparer_test.prepare_labels_for_yolo()

# data_preparer = TauPreparer(in_root_dir=in_root_dir, in_img_dir="images/train/747297kept", in_label_dir="labels/train/747297kept",
#                             prepared_root_dir=in_root_dir, prepared_img_dir="images/train", prepared_label_dir="labels/train")
# data_preparer.prepare_labels_for_yolo()
# data_preparer_test.separate_by_tiles_dict(tiles)

# import os
# import image_labelling
# import data_utils

# def visualize_class_images(root_dir, img_dir, label_dir, target_class=1):
#     """
#     Visualize all images containing a specific class.
    
#     Args:
#         root_dir (str): Root directory path
#         img_dir (str): Directory containing images
#         label_dir (str): Directory containing labels
#         target_class (int): Class index to visualize (1 for CB)
#     """
#     # Define color mapping
#     class_to_colour = {
#         0: "green",    # TA
#         1: "#19ffe8",  # CB (cyan)
#         2: "#a225f5",  # NFT (purple)
#         3: "red"       # tau_fragments
#     }
    
#     # Walk through all subdirectories in the image directory
#     for root, _, files in os.walk(os.path.join(root_dir, img_dir)):
#         for file in files:
#             if file.endswith('.png'):
#                 img_path = os.path.join(root, file)
#                 filename = data_utils.get_filename(img_path)
                
#                 # Construct corresponding label path
#                 relative_path = os.path.relpath(root, os.path.join(root_dir, img_dir))
#                 label_path = os.path.join(root_dir, label_dir, relative_path, filename + '.txt')
                
#                 if os.path.exists(label_path):
#                     # Read bounding boxes and labels
#                     bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path)
                    
#                     # Check if target class exists in this image
#                     if target_class in labels:
#                         print(f"Found CB in: {filename}")
                        
#                         # Get colors for each box based on class
#                         colours = [class_to_colour[ci] for ci in labels]
                        
#                         # Display the image with bounding boxes
#                         image_labelling.show_bboxes(img_path, bboxes, labels=labels, colours=colours)
#                         _ = input("Press Enter to continue...")

# # Usage example
# root_dir = "M:/Unused/TauCellDL"
# img_dir = "images/test"
# label_dir = "labels/test"

# visualize_class_images(root_dir, img_dir, label_dir, target_class=1)

import os
import cv2
import matplotlib.pyplot as plt
from image_labelling import bboxes_from_yolo_labels

def is_box_at_boundary(bbox, img_shape, threshold=5):
    """
    Check if a bounding box is at the image boundary.
    
    Args:
        bbox: List of [x0, y0, x1, y1] coordinates
        img_shape: Tuple of (height, width)
        threshold: Number of pixels from boundary to consider
    """
    x0, y0, x1, y1 = bbox
    height, width = img_shape
    
    return (x0 <= threshold or  # Left boundary
            y0 <= threshold or  # Top boundary
            x1 >= width - threshold or  # Right boundary
            y1 >= height - threshold)  # Bottom boundary

def visualize_boundary_objects(root_dir, img_dir, label_dir):
    """
    Visualize all images containing objects at the boundary.
    """
    # Define color mapping
    class_to_colour = {
        0: "green",    # TA
        1: "#19ffe8",  # CB (cyan)
        2: "#a225f5",  # NFT (purple)
        3: "red"       # tau_fragments
    }
    
    # Walk through all subdirectories in the test directory
    test_img_dir = os.path.join(root_dir, img_dir)
    test_label_dir = os.path.join(root_dir, label_dir)
    
    for root, _, files in os.walk(test_img_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, test_img_dir)
                label_path = os.path.join(test_label_dir, relative_path, os.path.splitext(file)[0] + '.txt')
                
                if os.path.exists(label_path):
                    # Read image and get dimensions
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_shape = img.shape[:2]
                    
                    # Read bounding boxes and labels
                    bboxes, labels = bboxes_from_yolo_labels(label_path)
                    
                    # Check if any box is at boundary
                    boundary_boxes = [i for i, bbox in enumerate(bboxes) if is_box_at_boundary(bbox, img_shape)]
                    
                    if boundary_boxes:
                        print(f"\nFound boundary objects in: {file}")
                        print(f"Number of boundary objects: {len(boundary_boxes)}")
                        
                        # Create figure
                        plt.figure(figsize=(10, 10))
                        plt.imshow(img)
                        
                        # Draw all boxes, highlighting boundary boxes
                        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                            color = class_to_colour[label]
                            linewidth = 3 if i in boundary_boxes else 1
                            x0, y0, x1, y1 = bbox
                            
                            plt.plot([x0, x1, x1, x0, x0], 
                                   [y0, y0, y1, y1, y0], 
                                   color=color, 
                                   linewidth=linewidth,
                                   label=f"Class {label}" if i in boundary_boxes else None)
                            
                            if i in boundary_boxes:
                                plt.text(x0, y0-5, f"Class {label}", 
                                       color=color, 
                                       bbox=dict(facecolor='white', alpha=0.7))
                        
                        plt.title("Objects at Boundary (thick lines)")
                        plt.axis('off')
                        plt.show()
                        
                        input("Press Enter to continue...")
                        plt.close()

# Usage
root_dir = "M:/Unused/TauCellDL"
img_dir = "images/test/747331"
label_dir = "labels/test/747331"

visualize_boundary_objects(root_dir, img_dir, label_dir)