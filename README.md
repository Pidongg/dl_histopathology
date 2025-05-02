# dl_histopathology

This repository contains tools and scripts for:
- Preparing custom tau histopathology slide dataset for deep learning tasks
- Training object detection models (YOLO11 and Faster R-CNN) on the histopathology images
- Evaluating model performance with traditional metrics (mAP, F1, confusion matrix) and PDQ
- Enabling custom model inference on QuPath software

## Project Structure

- **data_preparation/** - Scripts for processing and labeling histopathology data
- **train_model/** - Training scripts for YOLO11 and PyTorch Faster R-CNN models
- **evaluation/** - Tools for model evaluation and traditional performance metrics
- **qupath_scripts/** - QuPath integration scripts for data export and model inference
- **config/** - Configuration files for various model architectures
- **pdq_evaluation/** - Run PDQ analysis
- **vision/** - PyTorch model modifications for skipping NMS, enabling all class confidence in predictions, and using per-class confidence thresholds for NMS.
- **ultralytics/** - Similar modifications for YOLO11

To install all dependencies, install `requirements.txt`.