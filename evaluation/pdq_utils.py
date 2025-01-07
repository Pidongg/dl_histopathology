import numpy as np
import cv2

class DetectionInstance:
    """Holds information for a single detection instance."""
    def __init__(self, bbox, confidence, class_probs, pos_prob=1.0):
        """
        Args:
            bbox (list): [x1, y1, x2, y2] coordinates
            confidence (float): Detection confidence score
            class_probs (np.ndarray): Class probability distribution
            pos_prob (float): Positional probability (spatial confidence)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_list = class_probs
        self.pos_prob = pos_prob
        
    def calc_heatmap(self, img_size):
        """Calculate probabilistic segmentation heatmap for the detection.
        
        Args:
            img_size (tuple): (height, width) of the image
            
        Returns:
            np.ndarray: Heatmap of shape (height, width) with values between 0-1
        """
        heatmap = np.zeros(img_size, dtype=np.float32)
        x1, y1, x2, y2 = self.box
        x1_c, y1_c = np.ceil(self.box[0:2]).astype(np.int)
        x2_f, y2_f = np.floor(self.box[2:]).astype(np.int)
        # Even if the cordinates are integers, there should always be a range
        x1_f = x1_c - 1
        y1_f = y1_c - 1
        x2_c = x2_f + 1
        y2_c = y2_f + 1

        heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]),
                max(x1_f, 0):min(x2_c + 1, img_size[1])] = self.pos_prob
        if y1_f >= 0:
            heatmap[y1_f, max(x1_f, 0):min(x2_c + 1, img_size[1])] *= y1_c - y1
        if y2_c < img_size[0]:
            heatmap[y2_c, max(x1_f, 0):min(x2_c + 1, img_size[1])] *= y2 - y2_f
        if x1_f >= 0:
            heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]), x1_f] *= x1_c - x1
        if x2_c < img_size[1]:
            heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]), x2_c] *= x2 - x2_f
        return heatmap

class GroundTruthInstance:
    def __init__(self, bbox, class_label, segmentation_mask_dir):
        self.bbox = bbox
        self.class_label = class_label
        self.segmentation_mask_dir = segmentation_mask_dir
        # Fix: Unpack bbox and pass segmentation_mask_dir separately
        x1, y1, x2, y2 = bbox
        # TODO: LOOP through all the files in the directory
        self.segmentation_mask = compute_segmentation_mask_for_bbox(x1, y1, x2, y2, segmentation_mask_dir)

def compute_segmentation_mask_for_bbox(x1, y1, x2, y2, segmentation_img_path):
    """Extract segmentation mask for a given bounding box.
    
    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates
        segmentation_img_path (str): Path to segmentation mask image
        
    Returns:
        np.ndarray: Binary segmentation mask for the bbox region
    """
    # Read segmentation mask
    seg_img = cv2.imread(str(segmentation_img_path), cv2.IMREAD_GRAYSCALE)
    if seg_img is None:
        raise ValueError(f"Could not read segmentation mask: {segmentation_img_path}")
        
    # Extract bbox region
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    bbox_mask = seg_img[y1:y2+1, x1:x2+1]
    
    return bbox_mask