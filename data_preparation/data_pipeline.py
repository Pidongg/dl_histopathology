from data_preparation.dataset_preparation import TauPreparer
import os

in_root_dir = "M:/Unused/TauCellDL"

empty_tiles_required = {
    "747297": 11,
    "747309": 19,
    "747316": 71,
    "747337": 0,
    "747350": 0,
    "747352": 0,
    "747814": 60,
    "747818": 25,
    "771746": 38,
    "771791": 57
}

# names:
#   0: TA
#   1: CB
#   2: NFT
#   3: tau_fragments

data_preparer = TauPreparer(in_root_dir=in_root_dir, in_img_dir="test_images", in_label_dir="test_labels",
                            prepared_root_dir=in_root_dir, prepared_img_dir="test_images_new", prepared_label_dir="test_labels_new", empty_tiles_required=empty_tiles_required)
data_preparer.show_bboxes("", label_dir="M:/Unused/TauCellDL/labels/validation/747352", img_dir="M:/Unused/TauCellDL/images/validation/747352", ext="")
# data_preparer.show_bboxes("", "M:/Unused/TauCellDL/images/747297", "M:/Unused/TauCellDL/labels_new")

# data_preparer.prepare_labels_for_yolo()
# data_preparer.train_test_val_split(train=0.8, test=0, valid=0.2)

tiles = {
    "train": {
        "bg": [747297, 747814, 747818],
        "cortical": [771746, 771791],
        "dn": [747350, 747337]},
    "validation": {
        "bg": [747309],
        "cortical": [747316],
        "dn": [747352]},
    "test": {
        "bg": [703488, 747821],
        "cortical": [747331, 771747],
        "dn": [747335, 771913]
    }
}
# data_preparer.separate_by_tiles_dict(tiles)

# data_preparer.show_bboxes("", "M:/Unused/TauCellDL/images/747309", "M:/Unused/TauCellDL/labels_new/747309")

# data_preparer_test = TauPreparer(in_root_dir=in_root_dir, in_img_dir="test_images", in_label_dir="test_labels_new",
#                             prepared_root_dir=in_root_dir, prepared_img_dir="test_images", prepared_label_dir="test_labels_new")
# data_preparer_test.prepare_labels_for_yolo()

# data_preparer = TauPreparer(in_root_dir=in_root_dir, in_img_dir="images", in_label_dir="labels",
#                             prepared_root_dir=in_root_dir, prepared_img_dir="images", prepared_label_dir="labels")
# data_preparer_test.separate_by_tiles_dict(tiles)