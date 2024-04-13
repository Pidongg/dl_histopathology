from data_preparation.data_preparer import DataPreparer
import os


if __name__ == "__main__":
    data_preparer = DataPreparer(in_root_dir='./dataset/Amgad2019',
                                 prepared_root_dir="./prepared_dataset/Amgad2019",
                                 patch_w=512,
                                 patch_h=512,
                                 in_img_dir="images",
                                 in_mask_dir="masks",
                                 prepared_img_dir="images",
                                 prepared_mask_dir="masks")

    img_lists = data_preparer.train_test_val_split(0.8, 0.1, 0.1)

    # split data into patches
    for set_type in img_lists:
        data_preparer.split_into_patches(image_list=img_lists[set_type], set_type=set_type)

    # get labels from masks
    for set_type in img_lists:
        data_preparer.bboxes_from_all_masks(set_type, yolo=True)