import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shap
from ultralytics import YOLO
import argparse
import cv2
import os
import sys
from PIL import Image

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import existing label reading function
from data_preparation.image_labelling import bboxes_from_yolo_labels

# Custom modules for SHAP integration
class CastNumpy(torch.nn.Module):
    """Converts NumPy arrays to PyTorch tensors."""
    def __init__(self, device='cuda'):
        super(CastNumpy, self).__init__()
        self.device = device
        
    def forward(self, image):
        image = np.ascontiguousarray(image)
        # Normalize image values to 0-1
        image = image.astype(np.float32) / 255.0
        # Convert from HWC to CHW format (move channels to first dimension)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).to(self.device)
        if image.ndimension() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        image = image.half() if self.device == 'cuda' else image.float()
        return image

class OD2Score(torch.nn.Module):
    """Converts YOLOv8 outputs to a scalar score for a target object."""
    def __init__(self, target_class, target_box, conf_thresh=0.25):
        """
        Args:
            target_class: Class ID of the target object
            target_box: [x1, y1, x2, y2] of the target object
            conf_thresh: Confidence threshold
        """
        super(OD2Score, self).__init__()
        self.target_class = target_class
        self.target_box = torch.tensor(target_box).float()
        self.conf_thresh = conf_thresh
        
    def forward(self, results):
        # Initialize score tensor
        score = torch.tensor([0.0])
        
        if len(results) == 0:
            return score
            
        if hasattr(results[0], 'boxes'):
            # For YOLOv8+ format
            boxes = results[0].boxes  # Get all boxes for the first image
            if len(boxes) == 0:
                return score
                
            # Get predictions above confidence threshold
            mask = boxes.conf >= self.conf_thresh
            if not mask.any():
                return score
                
            # Filter by confidence threshold
            classes = boxes.cls[mask]
            confs = boxes.conf[mask]
            xyxy = boxes.xyxy[mask]
            
            # Filter by correct class
            class_mask = classes == self.target_class
            if not class_mask.any():
                return score
                
            xyxy = xyxy[class_mask]
            confs = confs[class_mask]
            
            # Calculate IoU with target box
            target_box = self.target_box.to(xyxy.device).unsqueeze(0)
            ious = box_iou(xyxy, target_box)  # Using torchvision's box_iou or equivalent
            
            if len(ious) > 0:
                # Get best matching detection
                best_iou, best_idx = ious.max(0)
                best_conf = confs[best_idx]
                
                # Final score is confidence * IoU
                score = best_conf * best_iou
        
        return score

def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes
    """
    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Calculate intersection area
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h

    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area.unsqueeze(1) + b2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-16)  # Add small epsilon to avoid division by zero
    
    return iou

class SuperPixler(torch.nn.Module):
    """Creates super pixels for SHAP analysis."""
    def __init__(self, image, super_pixel_size=16, mask_value=None):
        super(SuperPixler, self).__init__()
        self.image = image.copy()
        self.super_pixel_size = super_pixel_size
        
        # Calculate mean value for masking
        if mask_value is None:
            mask_value = np.mean(image, axis=(0, 1))
        self.mask_value = mask_value
        
        # Calculate super pixel dimensions
        h, w = image.shape[:2]
        self.h_super = h // super_pixel_size
        self.w_super = w // super_pixel_size
        
    def forward(self, mask):
        # mask is a binary array indicating which super pixels to keep
        # Ensure mask is the correct size
        if len(mask.shape) > 1:
            # If mask is 2D or higher, flatten it
            mask = mask.flatten()
        
        # Take only the number of elements we need
        needed_elements = self.h_super * self.w_super
        if len(mask) > needed_elements:
            mask = mask[:needed_elements]
        elif len(mask) < needed_elements:
            # Pad with ones if mask is too small
            mask = np.pad(mask, (0, needed_elements - len(mask)), constant_values=1)
            
        # Reshape to match super pixel grid
        mask = mask.reshape(self.h_super, self.w_super)
        
        # Create masked image
        masked_img = self.image.copy()
        
        # Apply masking
        for i in range(self.h_super):
            for j in range(self.w_super):
                if mask[i, j] == 0:  # If super pixel is masked
                    h_start = i * self.super_pixel_size
                    h_end = min((i + 1) * self.super_pixel_size, self.image.shape[0])
                    w_start = j * self.super_pixel_size
                    w_end = min((j + 1) * self.super_pixel_size, self.image.shape[1])
                    
                    masked_img[h_start:h_end, w_start:w_end] = self.mask_value
                    
        return masked_img

def run_shap_analysis(model_path, image_path, target_class, target_box, 
                      super_pixel_size=16, num_samples=1000, output_path=None,
                      confidence=None, gt_label_path=None, class_names=None):
    """
    Run SHAP analysis on a single image with a YOLOv8 model.
    
    Args:
        model_path: Path to trained model .pt file
        image_path: Path to image for analysis
        target_class: Class ID of the target object
        target_box: [x1, y1, x2, y2] of the target object
        super_pixel_size: Size of super pixels (default: 16)
        num_samples: Number of samples for SHAP analysis
        output_path: Path to save visualization (if None, will display)
        confidence: Confidence score of the detection
        gt_label_path: Path to ground truth label file (YOLO format)
        class_names: List of class names (for display purposes)
    """
    # Load model
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get original image dimensions to use for scaling bounding box
    orig_height, orig_width = image.shape[:2]
    
    # Resize image to be divisible by 32 (YOLO requirement)
    new_height = (orig_height // 32) * 32
    new_width = (orig_width // 32) * 32
    if new_height != orig_height or new_width != orig_width:
        scale_h = new_height / orig_height
        scale_w = new_width / orig_width
        # Scale the target box accordingly
        target_box = [
            target_box[0] * scale_w,
            target_box[1] * scale_h,
            target_box[2] * scale_w,
            target_box[3] * scale_h
        ]
        # Resize the image
        image = cv2.resize(image, (new_width, new_height))
    
    img_height, img_width = image.shape[:2]
    
    # Create components for SHAP analysis
    super_pixler = SuperPixler(image, super_pixel_size=super_pixel_size)
    numpy2torch = CastNumpy(device=device)
    od2score = OD2Score(target_class, target_box, conf_thresh=0.25)
    
    # Create sequential model for SHAP
    sequential_model = torch.nn.Sequential(
        super_pixler,
        numpy2torch,
        model,
        od2score
    )
    
    # Create wrapper function to convert PyTorch output to numpy
    def model_wrapper(mask_input):
        with torch.no_grad():
            if len(mask_input.shape) == 1:
                # Single input
                output = sequential_model(mask_input)
                return output.cpu().numpy()
            else:
                # Batch of inputs
                outputs = []
                for i, mask in enumerate(mask_input):
                    output = sequential_model(mask)
                    outputs.append(output.cpu().numpy())
                return np.array(outputs).reshape(-1, 1)
    
    # Calculate number of super pixels
    n_super_h = img_height // super_pixel_size
    n_super_w = img_width // super_pixel_size
    num_super_pixels = n_super_h * n_super_w
    
    print(f"Image size: {img_height}x{img_width}, Super pixel grid: {n_super_h}x{n_super_w}")
    
    # Create background (masked image)
    background_mask = np.zeros(num_super_pixels)
    
    # Create mask for analysis (all superpixels visible)
    mask = np.ones(num_super_pixels)
    
    # Initialize SHAP explainer
    print("Initializing SHAP explainer...")
    explainer = shap.KernelExplainer(model_wrapper, np.array([background_mask]))
    
    # Run SHAP analysis
    print(f"Running SHAP analysis with {num_samples} samples...")
    shap_values = explainer.shap_values(np.array([mask]), nsamples=num_samples)[0]
    
    # Prepare visualization
    shap_image = shap_values.reshape(n_super_h, n_super_w)
    shap_image = np.repeat(np.repeat(shap_image, super_pixel_size, axis=0), super_pixel_size, axis=1)
    
    # Normalize for visualization
    abs_max = np.max(np.abs(shap_values))
    normalized_shap = shap_image / (abs_max * 2) + 0.5
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    title = f"Super pixel contribution to detection\nClass: {target_class}"
    if confidence is not None:
        title += f", Confidence: {confidence:.3f}"
    plt.title(title)
    plt.imshow(image, alpha=0.7)
    plt.imshow(normalized_shap, cmap=plt.cm.seismic, alpha=0.5, vmin=0, vmax=1)
    
    # Add detection bounding box (green)
    rect = patches.Rectangle(
        (target_box[0], target_box[1]),
        target_box[2] - target_box[0],
        target_box[3] - target_box[1],
        linewidth=2, edgecolor='g', facecolor='none'
    )
    plt.gca().add_patch(rect)
    
    # Add detection class label
    class_name = f"Class {target_class}"
    if class_names and target_class < len(class_names):
        class_name = class_names[target_class]
    
    plt.text(target_box[0], target_box[1] - 10, 
             f"DET: {class_name}", 
             color='green', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Add ground truth boxes if provided (orange dashed boxes)
    if gt_label_path and os.path.exists(gt_label_path):
        try:
            # Use existing function to read YOLO format labels
            gt_boxes, gt_labels = bboxes_from_yolo_labels(gt_label_path, normalised=True)
            
            print(f"Found {len(gt_boxes)} ground truth boxes in {gt_label_path}")
            
            # Convert normalized coordinates to absolute coordinates
            for i, box in enumerate(gt_boxes):
                # Convert from PyTorch tensor to numpy if needed
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()
                
                # YOLO format is [x1, y1, x2, y2] in normalized coordinates
                x1, y1, x2, y2 = box
                
                # Convert to absolute coordinates
                x1 *= img_width
                y1 *= img_height
                x2 *= img_width
                y2 *= img_height
                
                # Draw ground truth box in orange with dashed line
                rect = patches.Rectangle(
                    (x1, y1), 
                    x2 - x1, 
                    y2 - y1,
                    linewidth=2, 
                    edgecolor='orange', 
                    linestyle='--',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)
                
                # Add class label
                class_id = gt_labels[i]
                if isinstance(class_id, torch.Tensor):
                    class_id = class_id.item()
                
                gt_class_name = f"Class {class_id}"
                if class_names and class_id < len(class_names):
                    gt_class_name = class_names[class_id]
                
                plt.text(x1, y1 - 10, 
                         f"GT: {gt_class_name}", 
                         color='orange', fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Error reading ground truth file: {e}")
    else:
        print(f"Ground truth file not found or not specified: {gt_label_path}")
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        print(f"Saved SHAP visualization to {output_path}")
    else:
        plt.show()
    
    return shap_values

def get_detections(model, image_path, conf_thresh=0.25):
    """
    Get all detections from an image above confidence threshold
    Returns: List of tuples (class_id, confidence, box)
    """
    results = model(image_path)
    detections = []
    
    if len(results) > 0 and hasattr(results[0], 'boxes'):
        boxes = results[0].boxes
        mask = boxes.conf >= conf_thresh
        
        if mask.any():
            classes = boxes.cls[mask].cpu().numpy()
            confs = boxes.conf[mask].cpu().numpy()
            xyxy = boxes.xyxy[mask].cpu().numpy()
            
            for cls, conf, box in zip(classes, confs, xyxy):
                detections.append({
                    'class_id': int(cls),
                    'confidence': float(conf),
                    'box': box.tolist()
                })
    
    return detections

def analyze_all_detections(model_path, image_path, super_pixel_size=16, 
                         num_samples=1000, conf_thresh=0.25, output_dir=None,
                         gt_label_dir=None, class_names=None):
    """
    Run SHAP analysis on all detections in an image
    
    Args:
        model_path: Path to trained model .pt file
        image_path: Path to image for analysis
        super_pixel_size: Size of super pixels
        num_samples: Number of samples for SHAP analysis
        conf_thresh: Confidence threshold for detections
        output_dir: Directory to save visualizations (if None, will display)
        gt_label_dir: Directory containing ground truth label files
        class_names: List of class names for display
    """
    # Load model
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Get all detections
    detections = get_detections(model, image_path, conf_thresh)
    print(f"Found {len(detections)} detections above confidence threshold {conf_thresh}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get corresponding ground truth label path if provided
    gt_label_path = None
    if gt_label_dir:
        # Extract image name without extension and path
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Try to find matching label file (.txt)
        possible_label_paths = [
            os.path.join(gt_label_dir, f"{img_basename}.txt"),
            os.path.join(gt_label_dir, img_basename, f"{img_basename}.txt"),
            os.path.join(gt_label_dir, f"{img_basename}.labels.txt")
        ]
        
        for path in possible_label_paths:
            if os.path.exists(path):
                gt_label_path = path
                print(f"Found ground truth label file: {gt_label_path}")
                break
                
        if not gt_label_path:
            print(f"Warning: No ground truth label file found for {img_basename}")
    
    # Run SHAP analysis for each detection
    for idx, detection in enumerate(detections):
        print(f"\nAnalyzing detection {idx + 1}/{len(detections)}")
        print(f"Class: {detection['class_id']}, Confidence: {detection['confidence']:.3f}")
        
        output_path = None
        if output_dir:
            filename = f"shap_class{detection['class_id']}_conf{detection['confidence']:.3f}_det{idx}.png"
            output_path = os.path.join(output_dir, filename)
        
        # Pass confidence to run_shap_analysis
        shap_values = run_shap_analysis(
            model_path=model_path,
            image_path=image_path,
            target_class=detection['class_id'],
            target_box=detection['box'],
            super_pixel_size=super_pixel_size,
            num_samples=num_samples,
            output_path=output_path,
            confidence=detection['confidence'],
            gt_label_path=gt_label_path,
            class_names=class_names
        )

def compare_shap_results(model_path, image_path, super_pixel_sizes, num_samples_list, 
                        conf_thresh=0.25, output_dir=None, gt_label_dir=None, class_names=None):
    """
    Run SHAP analysis with different super pixel sizes and sample numbers for comparison
    
    Args:
        model_path: Path to trained model .pt file
        image_path: Path to image for analysis
        super_pixel_sizes: List of super pixel sizes to test
        num_samples_list: List of sample numbers to test
        conf_thresh: Confidence threshold for detections
        output_dir: Directory to save visualizations
        gt_label_dir: Directory containing ground truth label files
        class_names: List of class names for display
    """
    # Load model
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Get all detections
    detections = get_detections(model, image_path, conf_thresh)
    print(f"Found {len(detections)} detections above confidence threshold {conf_thresh}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract image name from path to create a subdirectory
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        img_output_dir = os.path.join(output_dir, img_name)
        os.makedirs(img_output_dir, exist_ok=True)
    else:
        img_output_dir = None
    
    # Get corresponding ground truth label path if provided
    gt_label_path = None
    if gt_label_dir:
        # Extract image name without extension and path
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Try to find matching label file (.txt)
        possible_label_paths = [
            os.path.join(gt_label_dir, f"{img_basename}.txt"),
            os.path.join(gt_label_dir, img_basename, f"{img_basename}.txt"),
            os.path.join(gt_label_dir, f"{img_basename}.labels.txt")
        ]
        
        for path in possible_label_paths:
            if os.path.exists(path):
                gt_label_path = path
                print(f"Found ground truth label file: {gt_label_path}")
                break
                
        if not gt_label_path:
            print(f"Warning: No ground truth label file found for {img_basename}")
    
    # Run SHAP analysis for each combination
    for spx in super_pixel_sizes:
        for samples in num_samples_list:
            print(f"\nAnalyzing with super pixel size: {spx}, samples: {samples}")
            
            # Create subdirectory for this combination
            combo_dir = None
            if img_output_dir:
                combo_dir = os.path.join(img_output_dir, f"spx{spx}_samples{samples}")
                os.makedirs(combo_dir, exist_ok=True)
            
            # Run SHAP analysis for each detection
            for idx, detection in enumerate(detections):
                print(f"Processing detection {idx + 1}/{len(detections)}")
                print(f"Class: {detection['class_id']}, Confidence: {detection['confidence']:.3f}")
                
                output_path = None
                if combo_dir:
                    filename = f"shap_class{detection['class_id']}_conf{detection['confidence']:.3f}_det{idx}.png"
                    output_path = os.path.join(combo_dir, filename)
                
                try:
                    shap_values = run_shap_analysis(
                        model_path=model_path,
                        image_path=image_path,
                        target_class=detection['class_id'],
                        target_box=detection['box'],
                        super_pixel_size=spx,
                        num_samples=samples,
                        output_path=output_path,
                        confidence=detection['confidence'],
                        gt_label_path=gt_label_path,
                        class_names=class_names
                    )
                except Exception as e:
                    print(f"Error processing detection {idx} with spx={spx}, samples={samples}: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", help="Path to model .pt file")
    parser.add_argument("-img", help="Path to image file or directory")
    parser.add_argument("-spx", type=str, default="16,32,64", help="Comma-separated list of super pixel sizes")
    parser.add_argument("-samples", type=str, default="100,500,1000", help="Comma-separated list of sample numbers")
    parser.add_argument("-conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("-out", default=None, help="Output directory for visualizations")
    parser.add_argument("-gt", default=None, help="Directory containing ground truth label files (YOLO format)")
    parser.add_argument("-class_names", type=str, default=None, help="Comma-separated list of class names")
    
    args = parser.parse_args()
    
    # Parse super pixel sizes and sample numbers
    super_pixel_sizes = [int(x) for x in args.spx.split(',')]
    num_samples_list = [int(x) for x in args.samples.split(',')]
    
    # Parse class names if provided
    class_names = None
    if args.class_names:
        class_names = args.class_names.split(',')
    
    # Check if input is a directory or single file
    if os.path.isdir(args.img):
        # Process all images in directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for img_file in os.listdir(args.img):
            if img_file.lower().endswith(image_extensions):
                img_path = os.path.join(args.img, img_file)
                print(f"\nProcessing image: {img_file}")
                
                try:
                    compare_shap_results(
                        model_path=args.pt,
                        image_path=img_path,
                        super_pixel_sizes=super_pixel_sizes,
                        num_samples_list=num_samples_list,
                        conf_thresh=args.conf,
                        output_dir=args.out,
                        gt_label_dir=args.gt,
                        class_names=class_names
                    )
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
    else:
        # Process single image
        compare_shap_results(
            model_path=args.pt,
            image_path=args.img,
            super_pixel_sizes=super_pixel_sizes,
            num_samples_list=num_samples_list,
            conf_thresh=args.conf,
            output_dir=args.out,
            gt_label_dir=args.gt,
            class_names=class_names
        )

if __name__ == "__main__":
    main()