import torch
import os
import tqdm
from data_preparation import data_utils, image_labelling
from .metrics import ObjectDetectionMetrics
from abc import abstractmethod
from torchvision.io import read_image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class Evaluator:
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir):
        self.model = model
        # If test_imgs/test_labels are already lists of files, use them directly
        self.test_imgs = (test_imgs if isinstance(test_imgs, list) 
                         else data_utils.list_files_of_a_type(test_imgs, ".png", recursive=True))
        self.test_labels = (test_labels if isinstance(test_labels, list) 
                          else data_utils.list_files_of_a_type(test_labels, ".txt", recursive=True))
        self.device = device
        self.class_dict = class_dict
        self.save_dir = save_dir

        self.preds = []  # list of tensors, each tensor holding the predictions for one image
        self.gt = []  # list of tensors, each tensor holding the ground truth labels for one image

        self.__get_preds_and_labels()

        self.metrics = ObjectDetectionMetrics(save_dir=self.save_dir,
                                            idx_to_name=self.class_dict,
                                            detections=self.preds,
                                            ground_truths=self.gt,
                                            device=self.device)

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

        bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path, normalised=False)
        labels = torch.as_tensor(labels, dtype=torch.int32)  # convert from list to tensor
        labels = labels.unsqueeze(1)

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

    def confusion_matrix(self, conf_threshold=0.25, all_iou=False, plot=False, prefix=""):
        if not self.preds and not self.gt:
            raise Exception("No predictions and/or ground truths found")

        return self.metrics.get_confusion_matrix(conf_threshold, all_iou=all_iou, plot=plot, prefix=prefix)

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

class YoloEvaluator(Evaluator):
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir):
        super().__init__(model, test_imgs, test_labels, device, class_dict, save_dir)

    def infer_for_one_img(self, img_path):
        ground_truths = self.get_labels_for_image(img_path)  # (N, 5) where N = number of labels
        predictions = self.model(img_path, verbose=False, conf=0, device=self.device)[0].boxes.data

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
    def __init__(self, model, test_imgs, test_labels, device, class_dict, save_dir):
        super().__init__(model, test_imgs, test_labels, device, class_dict, save_dir)

    def infer_for_one_img(self, img_path):
        from torchvision.transforms import v2 as T

        self.model.eval()

        image = read_image(img_path)

        transforms = []
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        transforms = T.Compose(transforms)

        with torch.no_grad():
            x = transforms(image)
            # convert RGBA -> RGB and move to device
            x = x[:3, ...].to(self.device)
            predictions = self.model([x, ])
            pred = predictions[0]

        labels = pred['labels']

        i = labels != 0  # indices of non-background class predictions
        bboxes = pred['boxes'][i]
        scores = pred['scores'][i].unsqueeze(-1)
        labels = labels[i].unsqueeze(-1) - 1

        # predictions (n, 6) for n predictions
        predictions = torch.cat([bboxes, scores, labels], dim=-1)

        ground_truths = self.get_labels_for_image(img_path)  # (N, 5) where N = number of labels

        return ground_truths, predictions