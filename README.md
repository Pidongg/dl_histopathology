# Deep Learning for Histopathology Image Analysis

A comprehensive framework for developing, training and evaluating deep learning models for histopathological image analysis, with a focus on object detection in whole slide images.

## Overview

This repository contains tools and scripts for:
- Preparing histopathology slide data for deep learning tasks
- Training object detection models (YOLOv8 and Faster R-CNN) on histopathology images
- Evaluating model performance with various metrics
- Integrating deep learning models into QuPath for practical applications

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- QuPath (for slide processing and model integration)

See `requirements.txt` for a complete list of dependencies.

## Project Structure

- **data_preparation/** - Scripts for processing and labeling histopathology data
- **train_model/** - Training scripts for YOLOv8 and Faster R-CNN models
- **evaluation/** - Tools for model evaluation and performance metrics
- **qupath_scripts/** - QuPath integration scripts for data export and model inference
- **config/** - Configuration files for various model architectures
- **pdq_evaluation/** - Position-quality-duplicate (PDQ) evaluation metrics
- **vision/** - Computer vision utilities
- **ultralytics/** - YOLOv8 implementation

## Workflow

### 1. Data Preparation

Export data from QuPath:
```
# Run in QuPath's Automate script editor
export_detections.groovy          # Extract annotations
export_image_tiles.groovy         # Extract image tiles
```

Process dataset:
```
# For specific datasets
./data_preparation/run_data_prep_tau.py
./data_preparation/run_data_prep_bcss.py
```

### 2. Model Training

Train YOLOv8:
```
python ./train_model/yolo_scripts/run_train_yolo.py
```

Train Faster R-CNN:
```
python ./train_model/rcnn_scripts/run_train_rcnn.py
```

### 3. Model Evaluation

Evaluate model performance:
```
python ./evaluation/run_eval.py
```

### 4. Model Deployment

Use the `qupath_scripts/custom_model_inference.groovy` script to run trained models within QuPath.

## Dataset Structure

Datasets should include:
- `images/` directory for image tiles
- `masks/` directory for segmentation masks (if available)
- `labels/` directory for bounding box annotations in text format

For the TAU dataset, directories should be organized by brain region, with images further split by slide.

## License

See the `LICENSE` file for details.
