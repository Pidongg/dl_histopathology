import unittest
from ultralytics import YOLO
from evaluation.evaluate import *
from evaluation.old_metrics import *
import torch
from torch import Tensor


def get_testcase_from_model():
    device = (torch.device(f'cuda:{torch.cuda.current_device()}')
              if torch.cuda.is_available()
              else 'cpu')

    torch.set_default_device(device)
    model = YOLO("../models/YOLOv8/Tau/0_baseline/weights/best.pt").to(device)

    # initialise evaluator
    evaluator = Evaluator(model=model, test_imgs="../prepared_data/Tau/images/valid",
                          test_labels="../prepared_data/Tau/labels/valid",
                          device=device,
                          num_classes=5)

    # get predictions and labels
    ground_truths, predictions = evaluator.infer_for_one_img(
        "../prepared_data/Tau/images/valid/747297 [d=0.98892,x=94937,y=79747,w=506,h=506].png")

    return ground_truths, predictions


def get_preloaded_testcase():
    device = (torch.device(f'cuda:{torch.cuda.current_device()}')
              if torch.cuda.is_available()
              else 'cpu')

    torch.set_default_device(device)

    gt = Tensor([[112.3162, 114.3399, 124.4585, 130.5296, 4.0000],
                 [31.3676,  33.3913,  43.5099,  49.5810, 4.0000],
                 [488.7273, 223.6206, 504.9170, 235.7628, 3.0000]]).to(device)

    pred = Tensor([[2.8451e+02, 4.3293e+02, 2.9657e+02, 4.4753e+02, 3.6824e-01, 4.0000e+00],
                   [4.8946e+02, 2.2127e+02, 5.0431e+02, 2.3763e+02, 3.5004e-01, 3.0000e+00],
                   [2.9771e+01, 3.5601e+01, 4.3956e+01, 5.0079e+01, 2.8528e-01, 4.0000e+00]]).to(device)

    return gt, pred


class TestEvaluation(unittest.TestCase):
    def test_get_confusion_matrix(self):
        device = (torch.device(f'cuda:{torch.cuda.current_device()}')
                  if torch.cuda.is_available()
                  else 'cpu')

        gt, pred = get_preloaded_testcase()

        m = ObjectDetectionMetrics(save_dir="./test_output", idx_to_name=None, num_classes=5, detections=[pred],
                                   ground_truths=[gt], device=device)

        matrix = m.get_confusion_matrix(iou_threshold=0.5)

        print(matrix)

    def test_match_predictions(self):
        device = (torch.device(f'cuda:{torch.cuda.current_device()}')
                  if torch.cuda.is_available()
                  else 'cpu')

        gt, pred = get_preloaded_testcase()

        m = ObjectDetectionMetrics(save_dir="./test_output", idx_to_name=None, num_classes=5, detections=[pred],
                                         ground_truths=[gt], device=device)

        iou_threshold = Tensor([0.5, 0.7])

        detection_true, conf, detection_cls, gt_cls, gt_per_pred = m.match_predictions(iou_threshold=iou_threshold)

        print(gt_per_pred)

        # test results
        torch.testing.assert_close(detection_true, Tensor([[0., 1., 1.], [0., 0., 1.]]).to(device))
        torch.testing.assert_close(conf, Tensor([0.3682, 0.3500, 0.2853]).to(device), rtol=1e-5, atol=5e-5)
        torch.testing.assert_close(detection_cls, Tensor([4., 4., 4.]).to(device))
        torch.testing.assert_close(gt_cls, Tensor([4., 4., 4.]).to(device))
        torch.testing.assert_close(gt_per_pred, Tensor([[-1., 2., 1.], [-1., -1., 1.]]).to(device))

    def test_ap_per_class(self):
        device = (torch.device(f'cuda:{torch.cuda.current_device()}')
                  if torch.cuda.is_available()
                  else 'cpu')

        gt, pred = get_preloaded_testcase()

        idx_to_name = {
            0: "TA",
            1: "CB",
            2: "NFT",
            3: "tau_fragments",
            4: "non_tau"
        }

        m = ObjectDetectionMetrics(save_dir="./test_output", idx_to_name=idx_to_name, num_classes=5, detections=[pred],
                                         ground_truths=[gt], device=device)

        iou_threshold = Tensor([0.5, 0.7])

        r_curve, p_curve = m.ap_per_class(iou_threshold=iou_threshold, plot=True)

        print(r_curve.shape)
        print(r_curve)
        print(p_curve.shape)
        print(p_curve)


if __name__ == '__main__':
    unittest.main()
