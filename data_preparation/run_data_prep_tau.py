from dataset_preparation import TauPreparer
import argparse

# validation set: require no empty tiles to align with test set. Train set: number of empty tiles set to number of most minority class.
empty_tiles_required = {
    "747297": 11,
    "747309": 0,
    "747316": 0,
    "747337": 54,
    "747350": 29,
    "747352": 0,
    "747814": 60,
    "747818": 25,
    "771746": 38,
    "771791": 57
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir",
                        help="Path to input dataset root directory")
    parser.add_argument("out_dir",
                        help="Path to output directory into which to write prepared data")

    args = parser.parse_args()

    in_root_dir = args.in_dir
    prepared_root_dir = args.out_dir

    data_preparer = TauPreparer(in_root_dir=in_root_dir,
                                in_img_dir="images",
                                in_label_dir="labels",
                                prepared_root_dir=prepared_root_dir,
                                prepared_img_dir="images",
                                prepared_label_dir="labels")

    data_preparer.prepare_labels_for_yolo()

    data_preparer.show_bboxes("train")

    data_preparer.count_objects()
