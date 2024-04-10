from data_preparation import *


if __name__ == "__main__":
    # DATA PREPARATION
    IN_ROOT_DIR = "./dataset"
    OUT_ROOT_DIR = "./prepared_dataset"

    dataset_name = "Amgad2019"

    data_preparer = DataPreparer(os.path.join(IN_ROOT_DIR, dataset_name), os.path.join(OUT_ROOT_DIR, dataset_name))

    # split data into train/test/val
    img_lists = data_preparer.train_test_val_split(0.8, 0.1, 0.1, "images")  # keys are train, test, valid

    # split data into patches
    for set_type in img_lists:
        data_preparer.split_into_patches(os.path.join("images", set_type),
                                         os.path.join("masks", set_type),
                                         512,
                                         512,
                                         img_lists[set_type])

    # bboxes = data_preparer.bboxes_from_one_mask('/Users/kwanwynn.tan.21/Programming/dl_histopathology/prepared_dataset/Amgad2019/masks/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_0.png', yolo=True)
    # show_bboxes('/Users/kwanwynn.tan.21/Programming/dl_histopathology/prepared_dataset/Amgad2019/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_0.png', bboxes)
    # split_and_show_masks('/Users/kwanwynn.tan.21/Programming/dl_histopathology/prepared_dataset/Amgad2019/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_0.png',
    #                      '/Users/kwanwynn.tan.21/Programming/dl_histopathology/prepared_dataset/Amgad2019/masks/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_0.png')

    # convert masks to bboxes
    for set_type in img_lists:
        data_preparer.bboxes_from_multiple_masks(os.path.join("masks", set_type), yolo=True)

        # sanity check:
        assert len(os.listdir(os.path.join(OUT_ROOT_DIR, dataset_name, "images", set_type))) == len(os.listdir(os.path.join(OUT_ROOT_DIR, dataset_name, "masks", set_type)))
        assert len(os.listdir(os.path.join(OUT_ROOT_DIR, dataset_name, "bbox_labels", set_type))) == len(os.listdir(os.path.join(OUT_ROOT_DIR, dataset_name, "masks", set_type)))


