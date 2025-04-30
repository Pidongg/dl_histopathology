import os
import torch
import tqdm
from data_preparation import data_utils
from metrics import ObjectDetectionMetrics
from abc import abstractmethod
from torchvision.io import read_image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import json
from model_utils import enable_dropout, monte_carlo_predictions
from pdq_evaluation.read_files import  convert_tau_histopathology_predictions_to_rvc, LOGGER
from torchvision.transforms import v2 as T


class Evaluator:
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir, 
                 save_predictions=False, mc_dropout=False, num_samples=30, 
                 iou_thresh=0.6, conf_thresh=0.25, save_predictions_path=None, save_rvc=None):
        self.model = model
        self.mc_dropout = mc_dropout
        self.iou_thresh = iou_thresh
        self.save_predictions = save_predictions
        self.conf_thresh = conf_thresh

        # If test_imgs/test_labels are already lists of files, use them directly
        self.test_imgs = (test_imgs if isinstance(test_imgs, list) 
                         else data_utils.list_files_of_a_type(test_imgs, ".png", recursive=True))
        self.test_imgs.sort()
        self.test_labels = (test_labels if isinstance(test_labels, list) 
                          else data_utils.list_files_of_a_type(test_labels, ".txt", recursive=True))
        self.test_labels.sort()
        self.device = device

        self.model.eval()
        if self.mc_dropout:
            enable_dropout(self.model)
            predictions_dict = self._generate_mc_predictions(num_samples)
        else:
            preds, gt, predictions_dict = self.no_mc_predictions()
            self.metrics = ObjectDetectionMetrics(
                save_dir=save_dir,
                idx_to_name=class_dict,
                detections=preds,
                ground_truths=gt,
                device=self.device
            )

        if self.save_predictions:
            self._save_predictions(predictions_dict, save_predictions_path, save_rvc, class_dict)

    @abstractmethod
    def infer_for_one_img(self, img_path):
        """ Get predictions along with ground truth labels for a given image path. """
        pass

    def no_mc_predictions(self):
        """
        Runs inference on all the images found in `self.test_imgs` and returns predictions + matching labels.
        """
        preds = []
        gt = []

        print("Running inference on test set...")
        for i in tqdm.tqdm(range(len(self.test_imgs))):
            img_path = self.test_imgs[i]
            ground_truths, predictions = self.infer_for_one_img(img_path)
            
            # Ensure both are tensors
            if ground_truths is not None and not isinstance(ground_truths, torch.Tensor):
                ground_truths = torch.tensor(ground_truths, device=self.device)
            
            if predictions is not None and not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions, device=self.device)
                
            # Add to the lists
            preds.append(predictions)
            gt.append(ground_truths)
            
        return preds, gt

    def _generate_mc_predictions(self, num_samples):
        """Generate Monte Carlo predictions and return as dictionary."""
        results_dict = {}
        
        for img_file in self.test_imgs:
            try:
                # Get processed predictions
                output, all_scores, covariances = monte_carlo_predictions(
                    img_file,
                    self.conf_thresh,
                    self.iou_thresh,
                    num_samples
                )
                # Generate image key and predictions
                rel_path = os.path.basename(img_file)
                # Clean up path if needed
                if '/' in rel_path:
                    parts = rel_path.split('/')
                    if parts[-2] in parts[-1]:
                        rel_path = parts[-1]
                
                # Format predictions
                image_dets = []
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
                
                results_dict[rel_path] = image_dets
                
            except Exception as e:
                LOGGER.warning(f"Error processing image {img_file}: {e}")
                rel_path = os.path.basename(img_file)
                # Clean up path if needed
                if '/' in rel_path:
                    parts = rel_path.split('/')
                    if parts[-2] in parts[-1]:
                        rel_path = parts[-1]
                results_dict[rel_path] = []
        
        return results_dict
        
    def _save_predictions(self, predictions_dict, save_predictions_path, save_rvc, class_dict):
        """Save predictions to file."""
        with open(save_predictions_path, 'w') as f:
            json.dump(predictions_dict, f, indent=2)
        convert_tau_histopathology_predictions_to_rvc(save_predictions_path, save_rvc, class_dict)

    def get_confusion_matrix(self, conf_threshold=0.25, all_iou=False, plot=False, prefix="", class_conf_thresholds=None):
        if self.mc_dropout:
            raise Exception("Confusion matrix not available for MC Dropout mode")

        return self.metrics.get_confusion_matrix(conf_threshold, all_iou=all_iou, plot=plot, prefix=prefix, class_conf_thresholds=class_conf_thresholds)

    def get_ap_per_class(self, plot=False, plot_all=False, prefix=""):
        if self.mc_dropout:
            raise Exception("AP per class not available for MC Dropout mode")

        ap = self.metrics.ap_per_class(plot=plot, plot_all=plot_all, prefix=prefix)
        return ap

    def get_map50(self):
        if self.mc_dropout:
            raise Exception("mAP50 not available for MC Dropout mode")
        return self.metrics.get_map50()

    def get_map50_95(self):
        if self.mc_dropout:
            raise Exception("mAP50-95 not available for MC Dropout mode")
        return self.metrics.get_map50_95()

class YoloEvaluator(Evaluator):
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir):
        super().__init__(model, test_imgs, test_labels, device, class_dict, save_dir)

    def infer_for_one_img(self, img_path):
        ground_truths = data_utils.get_labels_for_image(self.device, img_path, self.test_imgs, self.test_labels)  # (N, 5) where N = number of labels
        if not self.mc_dropout:
            predictions = self.model(img_path, verbose=False, conf=0, device=self.device)[0].boxes.data
        else:
            predictions = monte_carlo_predictions(img_path, self.conf_thresh, self.iou_thresh, self.num_samples, skip_nms=True)
        # TODO: add prediction saving for YOLO
        # if self.save_predictions:
        #             all_scores = pred['scores_dist'].cpu().numpy() if 'scores_dist' in pred else None
        #             image_preds = []
        #             for j, (box, score, label) in enumerate(zip(bboxes, scores, labels)):
        #                 pred_dict = {
        #                     "boxes": [float(x) for x in box.cpu()],
        #                     "cls_id": int(label.item()),
        #                     "conf": float(score.item()),
        #                     "class_confs": [float(x) for x in all_scores[j][1:]] if all_scores is not None else [float(score.item())]
        #                 }
        #                 image_preds.append(pred_dict)
        #             self.predictions_dict[clean_name] = image_preds

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
        ground_truths = data_utils.get_labels_for_image(self.device, img_path, self.test_imgs, self.test_labels)
        
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
                ground_truths = data_utils.get_labels_for_image(self.device, img_path, self.test_imgs, self.test_labels)
                
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