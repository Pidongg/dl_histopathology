from ultralytics import YOLO
import torch
from evaluation import evaluate, metrics


def main():
    device = (torch.device(f'cuda:{torch.cuda.current_device()}')
              if torch.cuda.is_available()
              else 'cpu')

    torch.set_default_device(device)

    # Load a model
    model = YOLO("../models/YOLOv8/Tau/0_baseline/weights/best.pt")
    model.to(device)

    evaluator = evaluate.Evaluator(model,
                                   test_imgs="..\\prepared_data\\Tau\\images\\valid",
                                   test_labels="..\\prepared_data\\Tau\\labels\\valid",
                                   device=device,
                                   num_classes=5)

    evaluator.get_preds_and_labels()

    preds = evaluator.preds
    gt = evaluator.gt

    idx_to_name = {
        0: "TA",
        1: "CB",
        2: "NFT",
        3: "tau_fragments",
        4: "non_tau"
    }

    m = metrics.ObjectDetectionMetrics(save_dir="./eval_output", idx_to_name=idx_to_name, num_classes=5, detections=preds,
                               ground_truths=gt, device=device)

    # m.ap_per_class(plot=True)

    matrix = m.get_confusion_matrix(conf_threshold=0.25)
    print(matrix)

    print("mAP@50: ", m.get_map50())
    print("mAP@50-95: ", m.get_map50_95())


if __name__ == "__main__":
    main()

