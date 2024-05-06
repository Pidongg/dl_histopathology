from dataset_preparation import BCSSPreparer
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir",
                        help="Path to input dataset root directory")
    parser.add_argument("out_dir",
                        help="Path to output directory into which to write prepared data")

    args = parser.parse_args()

    in_root_dir = args.in_dir
    prepared_root_dir = args.out_dir

    data_preparer = BCSSPreparer(in_root_dir=in_root_dir,
                                 prepared_root_dir=prepared_root_dir,
                                 patch_w=512,
                                 patch_h=512,
                                 in_img_dir="images",
                                 in_mask_dir="masks",
                                 prepared_img_dir="images",
                                 prepared_mask_dir="masks")

    img_lists = data_preparer.get_train_test_val_img_lists(0.8, 0.1, 0.1)

    # split data into patches
    for set_type in img_lists:
        data_preparer.split_into_patches(image_list=img_lists[set_type], set_type=set_type)

    # get labels from masks
    for set_type in img_lists:
        data_preparer.bboxes_from_all_masks(set_type, yolo=True)

    data_preparer.show_masks_and_bboxes("train")

