import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F


def list_files_of_a_type(directory: os.path, file_extension: str) -> list[os.path]:
    """
    Lists all files with a specified file extension found in a directory.

    Args:
        directory (path): Path to the directory to look into.
        file_extension (str): e.g. ".png", ".jpg"

    Returns:
        (list[path]) A list of files with the given extension inside the directory.
    """
    return glob.glob(f"{directory}/*{file_extension}")


def get_filename(file_path):
    """ Get the filename of a file without the extension. """
    return os.path.splitext(os.path.basename(file_path))[0]


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
