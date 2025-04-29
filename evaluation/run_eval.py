import os

from ultralytics import YOLO
import torch
from evaluator import SAHIYoloEvaluator, YoloEvaluator, RCNNEvaluator
from data_preparation import data_utils
import argparse
import yaml
import train_model.rcnn_scripts.run_train_rcnn as run_train_rcnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-yolo", action="store_true",
                        help="Set flag to indicate a YOLO model is being passed")
    parser.add_argument("-rcnn", action="store_true",
                        help="Set flag to indicate an R-CNN model is being passed")
    parser.add_argument("model",
                        help="Path to model")
    parser.add_argument("cfg",
                        help="Path to a yaml file holding information about the dataset")
    parser.add_argument("save_dir",
                        help="Path to a directory to save output plots to")
    parser.add_argument("test_set_images",
                        help="Path to test set image directory")
    parser.add_argument("test_set_labels",
                        help="Path to test set label directory")
    parser.add_argument("-name",
                        help="(Optional) Identifier to prefix any output plots with")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to a file", default=False)
    parser.add_argument("--save_predictions_path", help="Name of the file to save predictions to", default="model_rcnn_yolo.json")
    parser.add_argument("--save_rvc", help="Path to save RVC predictions to", default="model_rcnn_rvc.json")
    
    # Add after other parser arguments
    parser.add_argument("-sahi", action="store_true",
                        help="Use SAHI tiled inference for evaluation",
                        default=False)
    parser.add_argument("--slice_size", type=int,
                        help="Slice size for SAHI tiled inference",
                        default=256)
    parser.add_argument("--overlap_ratio", type=float,
                        help="Overlap ratio for SAHI tiled inference",
                        default=0.2)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic architecture for R-CNN",
                        default=False)
    parser.add_argument("--num_samples",
                        type=int,
                        default=10,
                        help='Number of Monte Carlo Dropout samples')
    parser.add_argument("-iou",
                        help='iou threshold',
                        default=0.5)
    parser.add_argument("-conf",
                        help='confidence threshold',
                        type=float,
                        default=0.25)
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate", default=0.5)
    parser.add_argument("--class_conf_thresh", type=str, 
                        help="Comma-separated list of confidence thresholds for each class, e.g. '0.3,0.5,0.7' or '[0.3,0.5,0.7]'", 
                        default=None)
    parser.add_argument("--mc_dropout", action="store_true",
                        help="Use Monte Carlo Dropout for R-CNN",
                        default=False)
    args = parser.parse_args()

    model_path = args.model
    cfg = args.cfg
    save_dir = args.save_dir
    test_images = args.test_set_images
    test_labels = args.test_set_labels
    prefix = args.name
    stochastic = args.stochastic

    # Get list of all test images and labels
    test_images = data_utils.list_files_of_a_type(test_images, ".png", recursive=True)
    test_labels = data_utils.list_files_of_a_type(test_labels, ".txt", recursive=True)

    # check that the test sets are not empty
    if len(test_images) == 0:
        print("No images found in provided test set")
        return

    if len(test_labels) == 0:
        print("No labels found in provided test set")
        return

    if not args.yolo and not args.rcnn:
        print("At least one of -yolo or -rcnn must be specified to run evaluation.")
        return

    if args.yolo and args.rcnn:
        print("Only one flag among -yolo and -rcnn may be set to true when running evaluation.")
        return

    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    print(f"\nUsing device: {device}")

    # get class dictionary
    with open(cfg, "r") as stream:
        cfg_dict = yaml.safe_load(stream)

        if 'names' not in cfg_dict:
            raise Exception("Provided yaml file is expected to contain a 'names' field that holds a dictionary"
                            "of class indices to names.")

        class_dict = cfg_dict['names']

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # run inference with the provided model
    if args.yolo:
        model = YOLO(model_path)
        model.to(device)

        if args.sahi:
            evaluator = SAHIYoloEvaluator(model,
                                        test_imgs=test_images,
                                        test_labels=test_labels,
                                        device=device,
                                        class_dict=class_dict,
                                        save_dir=save_dir,
                                        slice_size=args.slice_size,
                                        overlap_ratio=args.overlap_ratio)
        else:
            evaluator = YoloEvaluator(model,
                                    test_imgs=test_images,
                                    test_labels=test_labels,
                                    device=device,
                                    class_dict=class_dict,
                                    save_dir=save_dir)  

    elif args.rcnn:
        num_classes = len(class_dict) + 1
        
        # Process class-specific confidence thresholds if provided
        class_specific_thresholds = None
        if args.class_conf_thresh:
            # Strip brackets if present and split by commas
            thresholds_str = args.class_conf_thresh.strip('[]')
            thresholds_list = [float(x) for x in thresholds_str.split(',')]
            
            # Convert list to dictionary with 1-indexed class IDs as keys
            # (since RCNN uses 0 as background class)
            if len(thresholds_list) == len(class_dict):
                class_specific_thresholds = {i+1: thresh for i, thresh in enumerate(thresholds_list)}
            else:
                print(f"Error: Number of provided thresholds ({len(thresholds_list)}) does not match number of classes ({len(class_dict)})")
                return
        
        model = run_train_rcnn.get_model_instance_segmentation(num_classes, stochastic=stochastic, 
                                                               all_scores=True, skip_nms=args.mc_dropout, 
                                                               conf_thresh=float(args.conf), 
                                                               iou_thresh=float(args.iou), 
                                                               dropout_rate=float(args.dropout_rate),
                                                               class_conf_thresh=class_specific_thresholds)
        
        # Load state dict to the correct device
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Force clear any previous output files
        if args.save_predictions and os.path.exists(args.save_predictions_path):
            os.remove(args.save_predictions_path)
        if args.save_rvc and os.path.exists(args.save_rvc):
            os.remove(args.save_rvc)
        evaluator = RCNNEvaluator(model,
                                  test_imgs=test_images,
                                  test_labels=test_labels,
                                  device=device,
                                  class_dict=class_dict,
                                  save_dir=save_dir,
                                  save_predictions=args.save_predictions,
                                  mc_dropout=args.mc_dropout,
                                  num_samples=args.num_samples,
                                  iou_thresh=float(args.iou),
                                  conf_thresh=float(args.conf),
                                  save_predictions_path=args.save_predictions_path,
                                  save_rvc=args.save_rvc,
                                  data_yaml=cfg)
        
    if args.save_predictions:
        return 
    ap = evaluator.ap_per_class(plot=True, plot_all=False, prefix=prefix)
    print("AP per class:", ap)
    matrix = evaluator.confusion_matrix(conf_threshold=0.25, all_iou=False, plot=True, prefix=prefix, class_conf_thresholds=class_specific_thresholds)
    print(matrix)

    print("mAP@50: ", evaluator.map50())
    print("mAP@50-95: ", evaluator.map50_95())


if __name__ == "__main__":
    main()