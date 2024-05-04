from data_preparation.dataset_preparation import TauPreparer
import os


if __name__ == "__main__":
    data_preparer = TauPreparer(in_root_dir='./datasets/Tau',
                                prepared_root_dir="./prepared_data/Tau",
                                patch_w=512,
                                patch_h=512,
                                in_img_dir="images",
                                in_label_dir="labels",
                                prepared_img_dir="images",
                                prepared_label_dir="labels")

    data_preparer.prepare_labels_for_yolo()

    data_preparer.train_test_val_split(train=0.8, test=0.1, valid=0.1,
                                       in_img_dir=os.path.join(data_preparer.in_root_dir, data_preparer.in_img_dir),
                                       in_label_dir=os.path.join(data_preparer.prepared_root_dir,
                                                                 data_preparer.prepared_label_dir))

    data_preparer.show_bboxes("train")


    # THE BOTTOM IS JUST TO VISUALISE ONE SINGLE SLIDE PLS PLS IT'S HARD CODED
    # from data_preparation import utils, image_labelling
    #
    # img_paths = utils.list_files_of_a_type("C:\\Users\\kwanw\\PycharmProjects\\dl_histopathology\\datasets\\Tau\\images\\Cortical\\747316",
    #                                        ".png")
    #
    # for img_path in img_paths:
    #     filename = utils.get_filename(img_path)
    #     print("viewing", filename)
    #
    #     label_path = os.path.join("C:\\Users\\kwanw\\PycharmProjects\\dl_histopathology\\prepared_data\\Tau\\labels\\Cortical\\747316", filename + ".txt")
    #     bboxes, labels = image_labelling.bboxes_from_yolo_labels(label_path)
    #     image_labelling.show_bboxes(img_path, bboxes, labels=labels)
    #     _ = input("enter to continue")