from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER
from confusion_matrix_utils import plot_confusion_matrix, save_interactive_confusion_matrix
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pdq_evaluation')))
from read_files import convert_tau_histopathology_predictions_to_rvc
from monte_carlo_dropout import save_mc_predictions_to_json
from data_preparation import data_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt",
                        help="Path to .pt file",
                        default='/local/scratch/pz286/dl_histopathology/weights/yolo11n.pt')
    parser.add_argument("-cfg",
                        help="Path to data config file",
                        default='/local/scratch/pz286/dl_histopathology/config/tau_data_test.yaml')
    parser.add_argument("-iou",
                        help='iou threshold',
                        default=0.5)
    parser.add_argument("-conf",
                        help='confidence threshold',
                        type=float,
                        default=0.25)
    parser.add_argument("-save_json",
                        help='Path to save YOLO format predictions JSON',
                        default='predictions_yolo.json')
    parser.add_argument("-save_rvc",
                        help='Path to save RVC format predictions JSON',
                        default='predictions_rvc.json')
    parser.add_argument("--save_detections",
                        action='store_true',
                        help='Whether to save detection results to JSON files')
    parser.add_argument("--save_interactive",
                        action='store_true',
                        help='Whether to save evaluation as interactive html')
    parser.add_argument("--save_interactive_path",
                        help='Path to save interactive HTML confusion matrix',
                        default='confusion_matrix.html')
    # Add Monte Carlo Dropout arguments
    parser.add_argument("--mc_dropout",
                        action='store_true',
                        help='Enable Monte Carlo Dropout for uncertainty estimation')
    parser.add_argument("--num_samples",
                        type=int,
                        default=10,
                        help='Number of Monte Carlo Dropout samples')
    parser.add_argument("--save_mc_predictions",
                        help='Path to save Monte Carlo predictions JSON',
                        default='predictions_mc.json')
    parser.add_argument("--input_size",
                        type=int,
                        default=640,
                        help='Input size for the model')
    parser.add_argument("--class_conf_thresholds", 
                    type=str,  # Accept as string initially
                    default=None)

    args = parser.parse_args()

    # Load model and config
    model = YOLO(args.pt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load class names
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    LOGGER.info(f"Loaded {len(class_names)} classes")
    if args.class_conf_thresholds is not None:
        # Strip brackets if present and split by commas
        thresholds_str = args.class_conf_thresholds.strip('[]')
        class_conf_thresholds = [float(x) for x in thresholds_str.split(',')]
    else:
        class_conf_thresholds = [float(args.conf)] * len(class_names)

    # Save predictions if requested
    if args.save_detections:
        if args.mc_dropout:
            LOGGER.info("Saving Monte Carlo Dropout predictions...")
            save_mc_predictions_to_json(
                model=model,
                data_yaml=args.cfg,
                conf_thresh=args.conf,
                save_path=args.save_mc_predictions,
                num_samples=args.num_samples,
                iou_thresh=float(args.iou),
                is_yolo=True,
                input_size=int(args.input_size),
                class_conf_thresholds=class_conf_thresholds
            )
            convert_tau_histopathology_predictions_to_rvc(args.save_mc_predictions, args.save_rvc, class_names)
            LOGGER.info(f"Saved Monte Carlo predictions to {args.save_mc_predictions}")
        else:
            LOGGER.info("Saving standard predictions to JSON files...")
            print(class_conf_thresholds)
            data_utils.save_predictions_to_json(model, args.cfg, args.conf, args.save_json, float(args.iou), class_conf_thresholds)
            convert_tau_histopathology_predictions_to_rvc(args.save_json, args.save_rvc, class_names)
            LOGGER.info("Finished saving predictions")

    if args.save_interactive:
        LOGGER.info("Creating interactive HTML confusion matrix...")
        save_interactive_confusion_matrix(model, args.cfg, class_names, args.save_interactive_path)
    else:
        metrics = model.val(
            data=args.cfg,
            conf=args.conf,
            iou=float(args.iou),
            class_conf_thresholds=class_conf_thresholds,
            imgsz = int(args.input_size)
        )

        # Print metrics
        LOGGER.info(f"mAP50-95: {metrics.box.map:.4f}")
        LOGGER.info(f"mAP50: {metrics.box.map50:.4f}")
        LOGGER.info(f"Per-class F1: {metrics.box.f1}")

        # Plot confusion matrix
        confusion_matrix = metrics.confusion_matrix
        plot_confusion_matrix(confusion_matrix, class_names)

if __name__ == "__main__":
    main()
          