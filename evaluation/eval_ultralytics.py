from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse
import torch
from ultralytics.utils import LOGGER
from confusion_matrix_utils import plot_confusion_matrix, save_interactive_confusion_matrix
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pdq_evaluation')))
from read_files import convert_yolo_to_rvc
from monte_carlo_dropout import save_mc_predictions_to_json
from data_preparation import data_utils

def save_predictions_to_json(model, data_yaml, conf_thresh, save_path, iou_thresh):
    """Save model predictions to JSON in YOLO format"""
    results_dict = {}
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    val_path = os.path.join(data_config.get('path', ''), data_config.get('val', ''))
    if not os.path.exists(val_path):
        LOGGER.error(f"Invalid validation path: {val_path}")
        return
    
    # Process all images in all subdirectories
    image_files = []
    for root, _, _ in os.walk(val_path):
        image_files.extend(data_utils.list_files_of_a_type(root, '.png'))
    
    # Sort files by path to ensure consistent ordering
    image_files.sort()
    
    # Run predictions on validation set
    for img_file in image_files:
        try:
            # Use model's training size (640) for inference
            results = model.predict(str(img_file), conf=conf_thresh, iou=iou_thresh)[0] # rescaling done by ultralytics
            boxes = results.boxes
            
            predictions = []
            if len(boxes) > 0:
                for box in boxes:
                    try:
                        xyxy = box.xyxy.cpu().numpy()
                        x1, y1, x2, y2 = xyxy.reshape(-1)[:4]
                        conf = float(box.data[0,4].cpu().numpy())
                        cls_id = int(box.data[0,5].cpu().numpy())
                        # Get all class confidences
                        class_confs = box.data[0, 6:].cpu().numpy()
                        
                        # Create prediction array
                        pred = {
                            'boxes': [float(x1), float(y1), float(x2), float(y2)],
                            'cls_id': cls_id,
                            'conf': conf,
                            'class_confs': class_confs.tolist()
                        }
                        predictions.append(pred)
                    except Exception as e:
                        LOGGER.warning(f"Error processing box: {e}")
                        continue
            
            # Use consistent path format
            rel_path = os.path.basename(str(img_file))
            results_dict[rel_path] = predictions
            
        except Exception as e:
            LOGGER.warning(f"Error processing image {img_file}: {e}")
            continue
    
    # Save to JSON
    try:
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    except Exception as e:
        LOGGER.error(f"Error saving predictions to {save_path}: {e}")

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
                input_size=int(args.input_size)
            )
            convert_yolo_to_rvc(args.save_mc_predictions, args.save_rvc, class_names)
            LOGGER.info(f"Saved Monte Carlo predictions to {args.save_mc_predictions}")
        else:
            LOGGER.info("Saving standard predictions to JSON files...")
            save_predictions_to_json(model, args.cfg, args.conf, args.save_json, float(args.iou))
            convert_yolo_to_rvc(args.save_json, args.save_rvc, class_names)
            LOGGER.info("Finished saving predictions")

    if args.save_interactive:
        LOGGER.info("Creating interactive HTML confusion matrix...")
        save_interactive_confusion_matrix(model, args.cfg, class_names, args.save_interactive_path)
    else:
        # Run validation with specified confidence threshold
        LOGGER.info(f"Running validation with confidence threshold {args.conf}...")
        metrics = model.val(
            data=args.cfg,
            conf=args.conf,
            iou=float(args.iou)
        )

        # Print metrics
        LOGGER.info(f"mAP50-95: {metrics.box.map:.4f}")
        LOGGER.info(f"mAP50: {metrics.box.map50:.4f}")
        LOGGER.info(f"mAP75: {metrics.box.map75:.4f}")
        LOGGER.info(f"Per-class mAP50-95: {metrics.box.maps}")

        # Plot confusion matrix
        confusion_matrix = metrics.confusion_matrix
        plot_confusion_matrix(confusion_matrix, class_names)

if __name__ == "__main__":
    main()
          