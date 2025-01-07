from data_preparation.dataset_preparation import TauPreparer

in_root_dir = "M:/Unused/TauCellDL"

# names:
#   0: TA
#   1: CB
#   2: NFT
#   3: tau_fragments

data_preparer = TauPreparer(in_root_dir=in_root_dir, in_img_dir="test_images_seg", in_label_dir="labels/test",
                            prepared_root_dir=in_root_dir, prepared_img_dir="test_images_new", prepared_label_dir="labels/test", with_segmentation=True, preprocessed_labels=True)

data_preparer.prepare_labels_for_yolo()