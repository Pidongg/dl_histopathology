import unittest
from ultralytics import YOLO
from evaluation.evaluator import *
import torch
from torch import Tensor


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

    def test_match_predictions(self):
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

        m = ObjectDetectionMetrics(save_dir="./test_output", idx_to_name=idx_to_name, detections=[pred],
                                   ground_truths=[gt], device=device)

        m.match_predictions()

        # test results
        torch.testing.assert_close(m.pred_true[0], Tensor([0, 1, 1]).int().to(device))
        torch.testing.assert_close(m.conf, Tensor([0.3682, 0.3500, 0.2853]).to(device), rtol=1e-5, atol=5e-5)
        torch.testing.assert_close(m.pred_cls, Tensor([4, 3, 4]).to(device))
        torch.testing.assert_close(m.gt_cls, Tensor([4, 4, 3]).to(device))
        torch.testing.assert_close(m.gt_per_pred_all[0], Tensor([-1, 2, 1]).int().to(device))

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

        m = ObjectDetectionMetrics(save_dir="./test_output", idx_to_name=idx_to_name,  detections=[pred],
                                   ground_truths=[gt], device=device)

        ap = m.ap_per_class()

        np.testing.assert_array_almost_equal(ap[0], np.array([1., 0.25]), decimal=1e-7)


if __name__ == '__main__':
    unittest.main()
