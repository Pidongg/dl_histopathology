import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

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
        x1, y1, x2, y2 (float): Bounding box coordinates
        segmentation_img_path (str): Path to segmentation mask image
        
    Returns:
        np.ndarray: Binary segmentation mask with same size as input image,
                   where regions outside bbox are set to 0
    """
    # Read segmentation mask and ensure it's a CPU numpy array
    seg_img = cv2.imread(str(segmentation_img_path), cv2.IMREAD_GRAYSCALE)
    if seg_img is None:
        raise ValueError(f"Could not read segmentation mask: {segmentation_img_path}")
    
    # Convert coordinates to numpy if they're tensors
    if torch.is_tensor(x1):
        x1 = x1.cpu().numpy()
    if torch.is_tensor(y1):
        y1 = y1.cpu().numpy()
    if torch.is_tensor(x2):
        x2 = x2.cpu().numpy()
    if torch.is_tensor(y2):
        y2 = y2.cpu().numpy()
    
    # Create empty mask of same size
    full_mask = np.zeros_like(seg_img)
    
    # Round coordinates to ensure we capture the full region
    x1_idx = int(np.floor(x1))
    y1_idx = int(np.floor(y1))
    x2_idx = int(np.ceil(x2)) +1
    y2_idx = int(np.ceil(y2)) +1
    
    # Ensure indices are within image bounds
    height, width = seg_img.shape
    x2_idx = min(width, x2_idx)
    y2_idx = min(height, y2_idx)
    
    # Extract bbox region and place it in the full mask
    bbox_region = seg_img[y1_idx:y2_idx, x1_idx:x2_idx]
    
    if bbox_region.size == 0 or np.all(bbox_region == 0):
        # visualize_segmentation(
        #     image_path=segmentation_img_path.replace('-labelled.png', '.png').replace('test_images_seg','images/test').replace('kept',''),
        #     segmentation_mask=bbox_region,
        #     bbox=[x1_idx, y1_idx, x2_idx, y2_idx],
        #     save_path='debug_extracted_region.png'
        # )
        raise ValueError(
            f"Ground truth instance must have non-zero segmentation mask within its bounding box. "
            f"Check the segmentation mask at {segmentation_img_path} for bbox coordinates "
            f"[{x1}, {y1}, {x2}, {y2}] (indexed as [{x1_idx}, {y1_idx}, {x2_idx}, {y2_idx}])"
            f"height: {height}, width: {width}"
        )
    
    full_mask[y1_idx:y2_idx, x1_idx:x2_idx] = bbox_region
    return full_mask

def visualize_segmentation(image_path, segmentation_mask, bbox, save_path=None):
    """
    Visualize segmentation mask overlaid on original image
    
    Args:
        image_path (str): Path to original image
        segmentation_mask (np.ndarray): Binary segmentation mask
        bbox (list): [x0, y0, x1, y1] coordinates
        save_path (str, optional): Path to save visualization
    """
    # Check if segmentation mask is valid
    if segmentation_mask is None or segmentation_mask.size == 0:
        print(f"Warning: Empty or invalid segmentation mask")
        return
        
    # Read original image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = image.copy()
    
    # Convert bbox coordinates to integers
    x0, y0, x1, y1 = map(int, bbox)
    
    # Extract region from image
    region = image[y0:y1, x0:x1]
    
    # Crop segmentation mask to match region size
    mask_cropped = segmentation_mask[y0:y1, x0:x1]
    
    # Create red overlay for the region
    mask_overlay = np.zeros_like(region)
    mask_overlay[mask_cropped > 0] = [255, 0, 0]  # Red color for mask
    
    try:
        # Blend mask with region
        region_overlay = cv2.addWeighted(
            region,
            0.5,  # alpha
            mask_overlay,
            0.5,  # beta = 1-alpha
            0     # gamma
        )
        
        # Place the overlay back into the full image
        overlay[y0:y1, x0:x1] = region_overlay
    except Exception as e:
        print(f"Warning: Segmentation mask shape {segmentation_mask.shape} doesn't match region shape {region.shape}")
        return
    
    # Draw bounding box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
    
    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()