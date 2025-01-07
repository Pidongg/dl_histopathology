# TODO: finish implementation

from pathlib import Path
import numpy as np
from .evaluator import Evaluator
from .pdq_utils import DetectionInstance, GroundTruthInstance, compute_segmentation_mask_for_bbox
from .pdq import PDQ

class PDQEvaluator(Evaluator):
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir, 
                 segmentation_dir=None):
        """
        Args:
            model: Detection model
            test_imgs: List of test image paths
            test_labels: List of label paths
            device: Computation device
            class_dict: Dictionary mapping class indices to names
            save_dir: Directory to save results
            segmentation_dir: Directory containing segmentation masks
        """
        self.segmentation_dir = Path(segmentation_dir) if segmentation_dir else None
        super().__init__(model, test_imgs, test_labels, device, class_dict, save_dir)
        self.pdq = PDQ()

    def get_segmentation_path(self, img_path):
        """Get corresponding segmentation mask path for an image."""
        if not self.segmentation_dir:
            raise ValueError("Segmentation directory not specified")
        
        img_name = Path(img_path).stem
        return self.segmentation_dir / f"{img_name}_mask.png"

    def create_detection_instances(self, predictions):
        """Convert raw predictions to DetectionInstance objects."""
        det_instances = []
        for pred in predictions:
            bbox = pred[:4].cpu().numpy()
            conf = pred[4].cpu().item()
            class_idx = int(pred[5].cpu().item())
            
            # Create one-hot class probabilities
            class_probs = np.zeros(len(self.class_dict))
            class_probs[class_idx] = conf
            
            det_instances.append(DetectionInstance(bbox, conf, class_probs))
        return det_instances

    def create_ground_truth_instances(self, ground_truths, img_path):
        """Convert ground truth labels to GroundTruthInstance objects."""
        gt_instances = []
        seg_path = self.get_segmentation_path(img_path)
        
        for gt in ground_truths:
            bbox = gt[:4].cpu().numpy()
            class_idx = int(gt[4].cpu().item())
            
            # Get segmentation mask for this bbox
            seg_mask = compute_segmentation_mask_for_bbox(*bbox, seg_path)
            
            gt_instances.append(GroundTruthInstance(bbox, class_idx, seg_mask))
        return gt_instances

    def infer_for_one_img(self, img_path):
        """Get predictions and ground truths, converting them to PDQ format."""
        ground_truths = self.get_labels_for_image(img_path)
        predictions = self.model(img_path, verbose=False, conf=0, device=self.device)[0].boxes.data
        
        # Convert to PDQ format
        det_instances = self.create_detection_instances(predictions)
        gt_instances = self.create_ground_truth_instances(ground_truths, img_path)
        
        # Calculate PDQ for this image
        self.pdq.add_img_eval(gt_instances, det_instances)
        
        return ground_truths, predictions

    def get_pdq_score(self):
        """Get the final PDQ score."""
        return self.pdq.get_pdq_score() 