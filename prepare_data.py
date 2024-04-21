from data_preparation.dataset_preparation import Amgad2019_Preparer, DeepLesion
import os


if __name__ == "__main__":
    # # Amgad2019 preparer:
    # data_preparer = Amgad2019(in_root_dir='./datasets/Amgad2019',
    #                              prepared_root_dir="./prepared_dataset/Amgad2019",
    #                              patch_w=512,
    #                              patch_h=512,
    #                              in_img_dir="images",
    #                              in_mask_dir="masks",
    #                              prepared_img_dir="images",
    #                              prepared_mask_dir="masks")
    #
    # img_lists = data_preparer.train_test_val_split(0.8, 0.1, 0.1)
    #
    # # split data into patches
    # for set_type in img_lists:
    #     data_preparer.split_into_patches(image_list=img_lists[set_type], set_type=set_type)
    #
    # # get labels from masks
    # for set_type in img_lists:
    #     data_preparer.bboxes_from_all_masks(set_type, yolo=True)

    # DeepLesion preparer:
    data_preparer = DeepLesion(in_root_dir=os.path.join('.', "datasets", "DeepLesion"),
                               in_img_dir="Key_slices",
                               prepared_root_dir=os.path.join('.', "prepared_data", "DeepLesion"),
                               prepared_img_dir="images",
                               prepared_label_dir="labels",
                               label_file="DL_info.csv")

    # data_preparer.train_test_val_split(0.8, 0.1, 0.1)
    # data_preparer.create_labels()
    # data_preparer.split_labels()

    data_preparer.process_extracted_images("C:\\Users\\kwanw\\PycharmProjects\\dl_histopathology\\datasets\\DeepLesion\\Images_png")

    # data_preparer.show_bboxes_from_labels("valid")
