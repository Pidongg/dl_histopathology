from ultralytics import YOLO
import torch
from evaluation.evaluator import YoloEvaluator, RCNNEvaluator
from data_preparation import data_utils
import argparse
import yaml
import run_train_rcnn


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

    args = parser.parse_args()

    model_path = args.model
    cfg = args.cfg
    save_dir = args.save_dir
    test_images = args.test_set_images
    test_labels = args.test_set_labels
    prefix = args.name

    # check that the test sets are not empty
    if len(data_utils.list_files_of_a_type(test_images, ".png")) == 0:
        print("No images found in provided test set")
        return

    if len(data_utils.list_files_of_a_type(test_labels, ".txt")) == 0:
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

    # run inference with the provided model
    if args.yolo:
        model = YOLO(model_path)
        # model = YOLO("./models/YOLOv8/Tau/0_baseline/weights/best.pt")
        model.to(device)

        # save_dir = "./eval_output"

        evaluator = YoloEvaluator(model,
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

    elif args.rcnn:
        # load model
        # num_classes = len(class_dict.keys())
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

        # print(evaluator.infer_for_one_img("./prepared_data/Tau/images/valid/747297 [d=0.98892,x=86867,y=37975,w=506,h=506].png"))


if __name__ == "__main__":
    main()

