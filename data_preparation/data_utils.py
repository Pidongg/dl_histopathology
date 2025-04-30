import os
import matplotlib.pyplot as plt
from data_preparation import image_labelling
import numpy as np
import torch
import torchvision.transforms.functional as F
import json
import yaml
import cv2

def list_files_of_a_type(directory, extension, recursive=False):
    """
    Lists all files with the given extension in the directory

    Args:
        directory: Path to search
        extension: File extension to look for (e.g., ".png")
        recursive: If True, search subdirectories as well
    """
    if recursive:
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return sorted(files)
    else:
        return sorted([os.path.join(directory, f) for f in os.listdir(directory)
                      if f.endswith(extension)])


def get_filename(file_path):
    """ Get the filename of a file without the extension. """
    tile = file_path.split('-labelled')[0]
    if not tile.endswith('.png'):
        tile = tile + '.png'
    return os.path.splitext(os.path.basename(tile))[0]


def show_images(images: list[F.Tensor]):
    """
    Takes images of type torch.Tensor, either a single image or a list, and plots them.
    Function code from https://pytorch.org/vision/0.11/auto_examples/plot_repurposing_annotations.html.
    """
    if not isinstance(images, list):
        images = [images]
    fix, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def cov(x):
    """Calculate covariance matrix for 2D coordinates."""
    x_centered = x - torch.mean(x, dim=0)
    return (x_centered.T @ x_centered) / (x.shape[0] - 1)


def is_pos_semidef(x):
    """Check if matrix is positive semi-definite."""
    return np.all(np.linalg.eigvals(x) >= -1e-8)


def get_near_psd(A):
    """Find the nearest positive semi-definite matrix to input matrix A."""
    B = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(B)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1] - 1)  # x1
    boxes[:, 1].clamp_(0, img_shape[0] - 1)  # y1
    boxes[:, 2].clamp_(0, img_shape[1] - 1)  # x2
    boxes[:, 3].clamp_(0, img_shape[0] - 1)  # y2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def save_predictions_to_json(model, data_yaml, conf_thresh, save_path, iou_thresh, class_conf_thresholds=None):
    """Save model predictions to JSON in YOLO format"""
    results_dict = {}

    # Load dataset config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    val_path = os.path.join(data_config.get(
        'path', ''), data_config.get('val', ''))
    if not os.path.exists(val_path):
        print(f"Invalid validation path: {val_path}")
        return

    # Process all images in all subdirectories
    image_files = []
    for root, _, _ in os.walk(val_path):
        image_files.extend(list_files_of_a_type(root, '.png'))

    # Sort files by path to ensure consistent ordering
    image_files.sort()

    # Run predictions on validation set
    for img_file in image_files:
        try:
            # Use model's training size (640) for inference
            results = model.predict(str(img_file), conf=conf_thresh, iou=iou_thresh,
                                    class_conf_thresholds=class_conf_thresholds)[0]  # rescaling done by ultralytics
            boxes = results.boxes

            predictions = []
            if len(boxes) > 0:
                for box in boxes:
                    try:
                        xyxy = box.xyxy.cpu().numpy()
                        x1, y1, x2, y2 = xyxy.reshape(-1)[:4]
                        conf = float(box.data[0, 4].cpu().numpy())
                        cls_id = int(box.data[0, 5].cpu().numpy())
                        # Get all class confidences
                        class_confs = box.data[0, 6:].cpu().numpy()

                        # Create prediction array
                        pred = {
                            'boxes': [float(x1), float(y1), float(x2), float(y2)],
                            'cls_id': cls_id,
                            'conf': conf,
                            'class_confs': class_confs.tolist()
                        }
                        predictions.append(pred)
                    except Exception as e:
                        print(f"Error processing box: {e}")
                        continue

            # Use consistent path format
            rel_path = os.path.basename(str(img_file))
            results_dict[rel_path] = predictions

        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

    # Save to JSON
    try:
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    except Exception as e:
        print(f"Error saving predictions to {save_path}: {e}")


def visualize_class_images(root_dir, img_dir, label_dir, target_class=1):
    """
    Visualize all images containing a specific class.

    Args:
        root_dir (str): Root directory path
        img_dir (str): Directory containing images
        label_dir (str): Directory containing labels
        target_class (int): Class index to visualize (1 for CB)
    """
    from data_preparation.image_labelling import bboxes_from_yolo_labels, show_bboxes

    # Define color mapping
    class_to_colour = {
        0: "green",    # TA
        1: "#19ffe8",  # CB (cyan)
        2: "#a225f5",  # NFT (purple)
        3: "red"       # tau_fragments
    }

    for root, _, files in os.walk(os.path.join(root_dir, img_dir)):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                filename = get_filename(img_path)

                # Construct corresponding label path
                relative_path = os.path.relpath(
                    root, os.path.join(root_dir, img_dir))
                label_path = os.path.join(
                    root_dir, label_dir, relative_path, filename + '.txt')

                if os.path.exists(label_path):
                    # Read bounding boxes and labels
                    bboxes, labels = bboxes_from_yolo_labels(label_path)

                    # Check if target class exists in this image
                    if target_class in labels:
                        print(f"Found CB in: {filename}")

                        # Get colors for each box based on class
                        colours = [class_to_colour[ci] for ci in labels]

                        # Display the image with bounding boxes
                        show_bboxes(img_path, bboxes,
                                    labels=labels, colours=colours)
                        _ = input("Press Enter to continue...")


def is_box_at_boundary(bbox, img_shape, threshold=5):
    """
    Check if a bounding box is at the image boundary.

    Args:
        bbox: List of [x0, y0, x1, y1] coordinates
        img_shape: Tuple of (height, width)
        threshold: Number of pixels from boundary to consider
    """
    x0, y0, x1, y1 = bbox
    height, width = img_shape

    return (x0 <= threshold or  # Left boundary
            y0 <= threshold or  # Top boundary
            x1 >= width - threshold or  # Right boundary
            y1 >= height - threshold)  # Bottom boundary


def visualize_boundary_objects(root_dir, img_dir, label_dir):
    """
    Visualize all images containing objects at the boundary.
    """
    # Import here to avoid circular imports
    from data_preparation.image_labelling import bboxes_from_yolo_labels

    # Define color mapping
    class_to_colour = {
        0: "green",    # TA
        1: "#19ffe8",  # CB (cyan)
        2: "#a225f5",  # NFT (purple)
        3: "red"       # tau_fragments
    }

    # Walk through all subdirectories in the test directory
    test_img_dir = os.path.join(root_dir, img_dir)
    test_label_dir = os.path.join(root_dir, label_dir)

    for root, _, files in os.walk(test_img_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, test_img_dir)
                label_path = os.path.join(
                    test_label_dir, relative_path, os.path.splitext(file)[0] + '.txt')

                if os.path.exists(label_path):
                    # Read image and get dimensions
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_shape = img.shape[:2]

                    # Read bounding boxes and labels
                    bboxes, labels = bboxes_from_yolo_labels(label_path)

                    # Check if any box is at boundary
                    boundary_boxes = [i for i, bbox in enumerate(
                        bboxes) if is_box_at_boundary(bbox, img_shape)]

                    if boundary_boxes:
                        print(f"\nFound boundary objects in: {file}")
                        print(
                            f"Number of boundary objects: {len(boundary_boxes)}")

                        # Create figure
                        plt.figure(figsize=(10, 10))
                        plt.imshow(img)

                        # Draw all boxes, highlighting boundary boxes
                        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                            color = class_to_colour[label]
                            linewidth = 3 if i in boundary_boxes else 1
                            x0, y0, x1, y1 = bbox

                            plt.plot([x0, x1, x1, x0, x0],
                                     [y0, y0, y1, y1, y0],
                                     color=color,
                                     linewidth=linewidth,
                                     label=f"Class {label}" if i in boundary_boxes else None)

                            if i in boundary_boxes:
                                plt.text(x0, y0-5, f"Class {label}",
                                         color=color,
                                         bbox=dict(facecolor='white', alpha=0.7))

                        plt.title("Objects at Boundary (thick lines)")
                        plt.axis('off')
                        plt.show()

                        input("Press Enter to continue...")
                        plt.close()


def get_labels_for_image(device, img_path, test_imgs, test_labels):
    """ Get the filename of a file without the extension. """
    filename = get_filename(img_path)

    # Get the subdirectory structure from the image path
    img_subdir = os.path.dirname(img_path)
    base_img_dir = os.path.dirname(os.path.commonprefix(test_imgs))
    relative_path = os.path.relpath(img_subdir, base_img_dir)

    # Construct the corresponding label path
    base_label_dir = os.path.dirname(os.path.commonprefix(test_labels))
    label_subdir = os.path.join(base_label_dir, relative_path)
    label_path = os.path.join(label_subdir, filename + ".txt")

    if not os.path.exists(label_path):
        raise Exception(f"Label file for {filename} not found at {label_path}")

    # Get the bounding boxes and labels
    bboxes, labels = image_labelling.bboxes_from_yolo_labels(
        label_path, normalised=False)

    # Ensure everything is on the correct device
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.to(device)
    else:
        bboxes = torch.tensor(bboxes, dtype=torch.float32, device=device)

    # Convert labels to tensor on the correct device
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device)
    else:
        labels = torch.tensor(labels, dtype=torch.int32, device=device)

    labels = labels.view(-1, 1)  # reshape to (n, 1)

    # Concatenate boxes and labels
    return torch.cat((bboxes, labels), dim=1)
