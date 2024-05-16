# dl_histopathology

Run `export_detections.groovy` and `export_image_tiles.groovy` in QuPath's Automate script editor to extract label data and tiles from slides.

Run `./data_preparation/run_data_prep_tau.py` or `./data_preparation/run_data_prep_bcss.py` (with the -h argument to see usage) to get data from the relevant dataset into the format needed for training.

Datasets should include a directory named 'images', a directory named 'masks' if masks are available, and a directory named 'labels' for bounding box text files. The tau dataset is expected to have each directory split by brain region, and for images, the directory for each region should be further split by slide.

Run `./yolo_scripts/run_train_yolo.py` to train a YOLOv8 model instance, and `./run_train_rcnn.py` to train a Faster R-CNN model instance.

Run `./run_eval.py` to calculate evaluation metrics for a given model and test set.
