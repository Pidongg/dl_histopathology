import torch
import os
import tqdm
from data_preparation import utils, image_labelling
from evaluation import old_metrics


class Evaluator:
    def __init__(self, model, test_imgs, test_labels, device, num_classes):
        self.model = model
        self.test_imgs = test_imgs  # path to the directory holding the test set images
        self.test_labels = test_labels  # path to the directory holding the test set labels
        self.device = device
        self.num_classes = num_classes

        self.preds = []  # list of tensors, each tensor holding the predictions for one image
        self.gt = []  # list of tensors, each tensor holding the ground truth labels for one image

    def infer_for_one_img(self, img_path):
        ground_truths = self.get_labels_for_image(img_path)  # (N, 5) where N = number of labels
        predictions = self.model(img_path, verbose=False, conf=0, device=self.device)[0].boxes.data

        return ground_truths, predictions

    def get_labels_for_image(self, img_path):
        filename = utils.get_filename(img_path)
        label_path = os.path.join(self.test_labels, filename + ".txt")

        if not os.path.exists(label_path):
            raise Exception(f"Label file for {filename} not found in label directory")

        bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path, normalised=False)
        labels = labels.unsqueeze(1)

        return torch.cat((bboxes, labels), axis=1)

    def get_preds_and_labels(self):
        """
        Runs inference on all the images found in `self.test_imgs` and stores predictions + matching labels in
        the relevant instance attributes.
        """
        # clear attributes to prevent double counting images
        self.preds = []
        self.gt = []

        img_paths = utils.list_files_of_a_type(self.test_imgs, ".png")

        print("Running inference on test set...")
        for i in tqdm.tqdm(range(len(img_paths))):
            img_path = img_paths[i]
            ground_truths, predictions = self.infer_for_one_img(img_path)

            self.preds.append(predictions)
            self.gt.append(ground_truths)

    def precision_and_recall(self, min_iou=0.5, min_conf=0.5):
        if not self.preds and not self.gt:
            raise Exception("No predictions and/or ground truths found")

        confusion_matrix = ConfusionMatrix(self.num_classes, iou_threshold=min_iou, conf_threshold=min_conf)

        print(f"Calculating precision and recall for IOU threshold {min_iou} and confidence threshold {min_conf}...")
        for i in tqdm.tqdm(range(len(self.preds))):
            predictions = self.preds[i]
            ground_truths = self.gt[i]

            confusion_matrix.update_confusion_matrix(detections=predictions, ground_truths=ground_truths)

        print(confusion_matrix.matrix)

        # 4. Output precision from this confusion matrix.
        return confusion_matrix.precision_per_class(), confusion_matrix.recall_per_class()

    def plot_pr_curve(self):
        pass
