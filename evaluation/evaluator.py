import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tqdm
from data_preparation import data_utils, image_labelling
from metrics import ObjectDetectionMetrics
from abc import abstractmethod
from torchvision.io import read_image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import json
from model_utils import enable_dropout, non_max_suppression
from pdq_evaluation.read_files import convert_yolo_to_rvc, LOGGER
import yaml
from torchvision.transforms import v2 as T


class Evaluator:
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir, save_predictions=False, mc_dropout=False, num_samples=30, iou_thresh=0.6, conf_thresh=0.25, save_predictions_path=None, save_rvc=None, data_yaml=None):
        self.model = model
        self.mc_dropout = mc_dropout
        self.num_samples = num_samples
        self.iou_thresh = iou_thresh
        self.save_predictions = save_predictions
        self.predictions_dict = {}
        self.save_predictions_path = save_predictions_path
        self.conf_thresh = conf_thresh
        self.data_yaml = data_yaml

        # If test_imgs/test_labels are already lists of files, use them directly
        self.test_imgs = (test_imgs if isinstance(test_imgs, list) 
                         else data_utils.list_files_of_a_type(test_imgs, ".png", recursive=True))
        self.test_imgs.sort()
        self.test_labels = (test_labels if isinstance(test_labels, list) 
                          else data_utils.list_files_of_a_type(test_labels, ".txt", recursive=True))
        self.test_labels.sort()
        self.device = device
        self.class_dict = class_dict
        self.save_dir = save_dir
        self.save_rvc = save_rvc

        self.preds = []  # list of tensors, each tensor holding the predictions for one image
        self.gt = []  # list of tensors, each tensor holding the ground truth labels for one image

        if self.mc_dropout:
            self.save_mc_predictions_to_json(self.data_yaml, self.conf_thresh, self.save_predictions_path, self.num_samples, self.iou_thresh)
            convert_yolo_to_rvc(self.save_predictions_path, self.save_rvc, self.class_dict)
        else:
            self.__get_preds_and_labels()

            self.metrics = ObjectDetectionMetrics(save_dir=self.save_dir,
                                            idx_to_name=self.class_dict,
                                            detections=self.preds,
                                            ground_truths=self.gt,
                                            device=self.device)
            if self.save_predictions:
                self.save_predictions_to_file(self.save_predictions_path)
                convert_yolo_to_rvc(self.save_predictions_path, self.save_rvc, self.class_dict)

    @abstractmethod
    def infer_for_one_img(self, img_path):
        """ Get predictions along with ground truth labels for a given image path. """
        pass

    def get_labels_for_image(self, img_path):
        """ Get the filename of a file without the extension. """
        filename = data_utils.get_filename(img_path)
        
        # Get the subdirectory structure from the image path
        img_subdir = os.path.dirname(img_path)
        base_img_dir = os.path.dirname(os.path.commonprefix(self.test_imgs))
        relative_path = os.path.relpath(img_subdir, base_img_dir)
        
        # Construct the corresponding label path
        base_label_dir = os.path.dirname(os.path.commonprefix(self.test_labels))
        label_subdir = os.path.join(base_label_dir, relative_path)
        label_path = os.path.join(label_subdir, filename + ".txt")

        if not os.path.exists(label_path):
            raise Exception(f"Label file for {filename} not found at {label_path}")

        # Get the bounding boxes and labels
        bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path, normalised=False)
        
        # Ensure everything is on the correct device
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.to(self.device)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32, device=self.device)
            
        # Convert labels to tensor on the correct device
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)
        else:
            labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
            
        labels = labels.view(-1, 1)  # reshape to (n, 1)
        
        # Concatenate boxes and labels
        return torch.cat((bboxes, labels), dim=1)

    def __get_preds_and_labels(self):
        """
        Runs inference on all the images found in `self.test_imgs` and stores predictions + matching labels in
        the relevant instance attributes.

        This should only be run once upon instantiation of the class.
        """
        # clear attributes to prevent double counting images
        self.preds = []
        self.gt = []

        print("Running inference on test set...")
        for i in tqdm.tqdm(range(len(self.test_imgs))):
            img_path = self.test_imgs[i]
            ground_truths, predictions = self.infer_for_one_img(img_path)
            self.preds.append(predictions)
            self.gt.append(ground_truths)

    def confusion_matrix(self, conf_threshold=0.25, all_iou=False, plot=False, prefix="", class_conf_thresholds=None):
        if not self.preds and not self.gt:
            raise Exception("No predictions and/or ground truths found")

        return self.metrics.get_confusion_matrix(conf_threshold, all_iou=all_iou, plot=plot, prefix=prefix, class_conf_thresholds=class_conf_thresholds)

    def ap_per_class(self, plot=False, plot_all=False, prefix=""):
        """
        ap (tensor[t, nc]): for t iou thresholds, nc classes
        """
        if not self.preds and not self.gt:
            raise Exception("No predictions and/or ground truths found")

        ap = self.metrics.ap_per_class(plot=plot, plot_all=plot_all, prefix=prefix)
        return ap

    def map50(self):
        return self.metrics.get_map50()

    def map50_95(self):
        return self.metrics.get_map50_95()
    
    def monte_carlo_predictions(self, img, conf_thresh, iou_thresh, num_samples=30):
        """
        Perform multiple forward passes with dropout enabled.
        Returns raw model outputs before NMS.
        """
        infs_all = []
        num_classes = 4  # RVC format: 4 classes (not including background)

        # Perform multiple forward passes
        with torch.no_grad():
            for _ in range(num_samples):
                # Get raw model output
                ground_truths, predictions = self.infer_for_one_img(img)
                if isinstance(predictions, list):
                    predictions = predictions[0]
                # For RCNN predictions, extract boxes, scores, and labels
                # if isinstance(predictions, dict):
                #     i = predictions['labels'] != 0  # indices of non-background class predictions
                #     boxes = predictions['boxes'][i]
                #     scores = predictions['scores'][i]
                #     labels = predictions['labels'][i] - 1  # Adjust class indices
                    
                #     # Create scores tensor for all classes
                #     class_scores = torch.zeros((len(boxes), num_classes), device=boxes.device)
                #     if 'scores_dist' in predictions:
                #         # If we have distribution scores, use them (excluding background class)
                #         class_scores = predictions['scores_dist'][i, 1:]  # Skip background class
                #     else:
                #         # Otherwise, put the score in the predicted class column
                #         class_scores[range(len(boxes)), labels] = scores
                    
                #     # Combine boxes and class scores: (N, 8) where 8 = 4 box coords + 4 class scores
                #     predictions = torch.cat([boxes, class_scores], dim=1)
                #     predictions = predictions.unsqueeze(0)  # Add batch dimension
                
                infs_all.append(predictions.unsqueeze(2))
            
            if num_samples > 1:
                inf_mean = torch.mean(torch.stack(infs_all), dim=0)
                infs_all.insert(0, inf_mean)
                inf_out = torch.cat(infs_all, dim=2)
            else:
                inf_out = infs_all[0]

            # Apply NMS and get sampled coordinates
            output, all_scores, sampled_coords = non_max_suppression(
                inf_out,
                conf_thres=conf_thresh,
                iou_thres=iou_thresh,
                multi_label=False,
                max_width=800,
                max_height=800
            )
            image_width = 512  # Or get actual image dimensions
            scale_factor = image_width / 800

            # Process sampled coordinates if we have multiple samples
            if num_samples > 1 and output[0] is not None and len(output[0]) > 0:
                # Use fixed dimensions
                height, width = 512, 512
                output[0][:, :4] *= scale_factor
                
                # Process sampled coordinates for the first (and only) image in batch
                sampled_boxes = data_utils.xywh2xyxy(sampled_coords[0].reshape(-1, 4)).reshape(sampled_coords[0].shape)
                sampled_boxes *= scale_factor
                data_utils.clip_coords(sampled_boxes.reshape(-1, 4), (height, width))

                # Calculate covariances for each detection
                covariances = torch.zeros(sampled_boxes.shape[0], 2, 2, 2, device=output[0].device)
                for det_idx in range(sampled_boxes.shape[0]):
                    covariances[det_idx, 0, ...] = data_utils.cov(sampled_boxes[det_idx, :, :2]) # covariance of top left corner
                    covariances[det_idx, 1, ...] = data_utils.cov(sampled_boxes[det_idx, :, 2:]) # covariance of bottom right corner

                # Ensure covariances are positive semi-definite
                for det_idx in range(len(covariances)):
                    for i in range(2):
                        cov_matrix = covariances[det_idx, i].cpu().numpy()
                        if not data_utils.is_pos_semidef(cov_matrix):
                            LOGGER.warning('Converting covariance matrix to near PSD')
                            cov_matrix = data_utils.get_near_psd(cov_matrix)
                            covariances[det_idx, i] = torch.tensor(cov_matrix, device=output[0].device)
                
                # Round values for smaller size
                covariances = torch.round(covariances * 1e6) / 1e6
            elif output[0] is not None and len(output[0]) > 0:
                # If we only have one sample but still have valid output, apply scaling
                output[0][:, :4] *= scale_factor
                covariances = None
            else:
                covariances = None
        return output, all_scores, covariances

    def save_mc_predictions_to_json(self, data_yaml, conf_thresh, save_path, num_samples=30, iou_thresh=0.6):
        """Save Monte Carlo predictions to JSON in same format as YOLO predictions."""
        results_dict = {}
        
        for img_file in self.test_imgs:
            try:
                # Get processed predictions
                output, all_scores, covariances = self.monte_carlo_predictions(
                    img_file,
                    conf_thresh,
                    iou_thresh,
                    num_samples
                )
                # Get relative path for image key
                rel_path = os.path.basename(img_file)
            
                # If the filename starts with a directory that matches its name, remove it
                # e.g., "747331/747331 [...]" becomes "747331 [...]"
                if '/' in rel_path:
                    parts = rel_path.split('/')
                    if parts[-2] in parts[-1]:
                        rel_path = parts[-1]
                image_dets = []
                # Format predictions for current image
                if output is not None and len(output) > 0:
                    output = output[0]
                    all_scores = all_scores[0]
                    for idx, pred in enumerate(output):
                        box = pred[:4].cpu().numpy()
                        conf = float(pred[4].cpu().numpy())
                        cls_id = int(pred[5].cpu().numpy())
                        class_confs = all_scores[idx].cpu().numpy()
                        
                        det_dict = {
                            "boxes": box.tolist(),
                            "conf": conf,
                            "cls_id": cls_id,
                            "class_confs": class_confs.tolist()
                        }
                        if covariances is not None:
                            det_dict["covars"] = covariances[idx].cpu().numpy().tolist()
                        image_dets.append(det_dict)
                
                # Add predictions for this image to results dict
                results_dict[rel_path] = image_dets
                
            except Exception as e:
                LOGGER.warning(f"Error processing image {img_file}: {e}")
                rel_path = os.path.basename(img_file)
            
                # If the filename starts with a directory that matches its name, remove it
                # e.g., "747331/747331 [...]" becomes "747331 [...]"
                if '/' in rel_path:
                    parts = rel_path.split('/')
                    if parts[-2] in parts[-1]:
                        rel_path = parts[-1]
                results_dict[rel_path] = []
                continue
    
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
        return results_dict
    
    def save_predictions_to_file(self, output_path):
        """Save accumulated predictions to a JSON file"""
        if not self.predictions_dict:
            print("No predictions to save. Run inference with save_predictions=True first.")
            return
            
        with open(output_path, 'w') as f:
            json.dump(self.predictions_dict, f, indent=2)
        
        print(f"Predictions saved to {output_path}")

class YoloEvaluator(Evaluator):
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir):
        super().__init__(model, test_imgs, test_labels, device, class_dict, save_dir)

    def infer_for_one_img(self, img_path):
        ground_truths = self.get_labels_for_image(img_path)  # (N, 5) where N = number of labels
        if not self.mc_dropout:
            predictions = self.model(img_path, verbose=False, conf=0, device=self.device)[0].boxes.data
        else:
            self.model.eval()
            enable_dropout(self.model)
            predictions = self.monte_carlo_predictions(img_path, self.conf_thresh, self.iou_thresh, self.num_samples, skip_nms=True)

        return ground_truths, predictions

class SAHIYoloEvaluator(YoloEvaluator):
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir, slice_size=256, overlap_ratio=0.2):
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        # Create SAHI detection model
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model.ckpt_path,
            confidence_threshold=0.0,  # Set to 0 to get all predictions
            device=str(device)  # Convert device to string format
        )
        super().__init__(model, test_imgs, test_labels, device, class_dict, save_dir)

    def infer_for_one_img(self, img_path):
        ground_truths = self.get_labels_for_image(img_path)
        
        # Use SAHI's sliced prediction
        result = get_sliced_prediction(
            img_path,
            self.sahi_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            verbose=0
        )
        
        # Convert SAHI predictions to the expected format
        predictions = torch.zeros((len(result.object_prediction_list), 6), device=self.device)
        for idx, pred in enumerate(result.object_prediction_list):
            bbox = pred.bbox.to_xyxy()
            predictions[idx] = torch.tensor([
                bbox[0], bbox[1], bbox[2], bbox[3],
                pred.score.value,
                pred.category.id
            ], device=self.device)
        
        return ground_truths, predictions

class RCNNEvaluator(Evaluator):
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir, save_predictions=False, mc_dropout=False, num_samples=30, iou_thresh=0.6, conf_thresh=0.25, save_predictions_path=None, save_rvc=None, data_yaml=None):
        super().__init__(model, test_imgs, test_labels, device, class_dict, save_dir, save_predictions, mc_dropout, num_samples, iou_thresh, conf_thresh, save_predictions_path, save_rvc, data_yaml)

    def infer_for_one_img(self, img_path):
        self.model.eval()  # Keep this to maintain eval mode for other layers

        # Enable dropout if we're doing MC dropout
        if self.mc_dropout:
            enable_dropout(self.model)

        # Get clean filename if we're saving predictions
        if self.save_predictions:
            base_name = os.path.basename(img_path)
            parts = base_name.split('[')
            if len(parts) > 1:
                clean_name = parts[0] + '[' + parts[1].split(']')[0] + '].png'
            else:
                clean_name = base_name

        # Load and transform image
        try:
            image = read_image(img_path)
        except Exception as e:
            print(f"Error reading image {img_path}: {str(e)}")
            # Return empty tensors if image can't be read
            return torch.zeros((0, 5), device=self.device), torch.zeros((0, 6), device=self.device)

        transforms = []
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        transforms = T.Compose(transforms)

        with torch.no_grad():
            try:
                # Ensure model is on the correct device
                self.model = self.model.to(self.device)
                
                # Transform and move image to device
                x = transforms(image)
                # convert RGBA -> RGB and move to device
                x = x[:3, ...].to(self.device)
                
                # Get ground truth with device parameter
                ground_truths = self.get_labels_for_image(img_path)
                
                # Get predictions
                predictions = self.model([x, ])
                
                if self.mc_dropout:
                    return ground_truths, predictions
                    
                # Check if predictions list is empty
                if not predictions:
                    print(f"Model returned empty predictions for {img_path}")
                    return ground_truths, torch.zeros((0, 6), device=self.device)
                
                pred = predictions[0]

                # Process prediction results
                labels = pred['labels']
                i = labels != 0  # indices of non-background class predictions
                
                # Ensure all tensors are on the same device
                bboxes = pred['boxes'][i].to(self.device)
                scores = pred['scores'][i].unsqueeze(-1).to(self.device)
                labels = labels[i].unsqueeze(-1).to(self.device) - 1  # Adjust class indices

                # If saving predictions, store them in the dictionary
                if self.save_predictions:
                    all_scores = pred['scores_dist'].cpu().numpy() if 'scores_dist' in pred else None
                    image_preds = []
                    for j, (box, score, label) in enumerate(zip(bboxes, scores, labels)):
                        pred_dict = {
                            "boxes": [float(x) for x in box.cpu()],
                            "cls_id": int(label.item()),
                            "conf": float(score.item()),
                            "class_confs": [float(x) for x in all_scores[j][1:]] if all_scores is not None else [float(score.item())]
                        }
                        image_preds.append(pred_dict)
                    self.predictions_dict[clean_name] = image_preds

                # Concatenate tensors that are guaranteed to be on the same device
                predictions = torch.cat([bboxes, scores, labels], dim=-1)

                return ground_truths, predictions

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                return torch.zeros((0, 5), device=self.device), torch.zeros((0, 6), device=self.device)