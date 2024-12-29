from ultralytics import YOLO
import torch
from evaluation.evaluator import SAHIYoloEvaluator, YoloEvaluator, RCNNEvaluator
from data_preparation import data_utils
import argparse
import yaml
import run_train_rcnn
import os


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
    args = parser.parse_args()

    model_path = args.model
    cfg = args.cfg
    save_dir = args.save_dir
    test_images = args.test_set_images
    test_labels = args.test_set_labels
    prefix = args.name

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

    device = (torch.device(f'cuda:{torch.cuda.current_device()}')
              if torch.cuda.is_available()
              else 'cpu')

    torch.set_default_device(device)

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
        # load model
        num_classes = len(class_dict) + 1
        model = run_train_rcnn.get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        evaluator = RCNNEvaluator(model,
                                  test_imgs=test_images,
                                  test_labels=test_labels,
                                  device=device,
                                  class_dict=class_dict,
                                  save_dir=save_dir)

        evaluator.ap_per_class(plot=True, plot_all=False, prefix=prefix)

        matrix = evaluator.confusion_matrix(conf_threshold=0.25, all_iou=False, plot=True)
        print(matrix)

        print("mAP@50: ", evaluator.map50())
        print("mAP@50-95: ", evaluator.map50_95())


if __name__ == "__main__":
    main()