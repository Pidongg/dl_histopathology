from data_preparation.dataset_preparation import TauPreparer
import os

in_root_dir = "M:/Unused/TauCellDL"

data_preparer = TauPreparer(in_root_dir=in_root_dir, in_img_dir="images", in_label_dir="labels/Training",
                            prepared_root_dir=in_root_dir, prepared_img_dir="images_new", prepared_label_dir="labels_new")
data_preparer.prepare_labels_for_yolo()
# data_preparer.train_test_val_split(train=0.8, test=0.1, valid=0.1)
