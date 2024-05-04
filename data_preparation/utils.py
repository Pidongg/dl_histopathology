import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F


def list_files_of_a_type(directory: os.path, file_extension: str) -> list[os.path]:
    """
    :param directory: Path to a directory
    :param file_extension: e.g. ".png", ".jpg"
    :return: A list of files with the given extension inside the directory.
    """
    return glob.glob(f"{directory}/*{file_extension}")


def get_filename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def show_images(images):
    """ Takes images (singular or a list) of type Image Tensor (Torch) and plots them. """
    if not isinstance(images, list):
        images = [images]
    fix, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()



