import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F


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
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y