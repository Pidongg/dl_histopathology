import glob
import os
import matplotlib.pyplot as plt
import numpy as np
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
