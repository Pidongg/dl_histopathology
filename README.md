# dl_histopathology

Run `./data_preparation/run_data_prep_tau.py` or `./data_preparation/run_data_prep_bcss.py` to get data into the format needed for training.

Datasets should include a directory named 'images', a directory named 'masks' if masks are available, and a directory named 'labels' for bounding box text files.

Run `./yolo_scripts/run_train_yolo.py` to train a YOLOv8 model instance, and `./run_train_rcnn.py` to train a Faster R-CNN model instance.

Run `./run_eval.py` to calculate evaluation metrics for a given model and test set.

Trained models are available in the `./models` directory.