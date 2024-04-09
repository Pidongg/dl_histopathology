# written with reference to `prepare_dataset.py` at https://github.com/nauyan/NucleiSegmentation

from skimage.util.shape import view_as_windows
from skimage.util import img_as_ubyte

import os
import glob
from skimage import io

from tqdm import tqdm


def split_into_patches(in_dir, out_dir):
    img_dir = "images"
    mask_dir = "masks"

    # make sure both dims are divisible by 32
    patch_width = 512
    patch_height = 512

    image_list = glob.glob(os.path.join(in_dir, img_dir) + "/*.png")

    out_img_dir = os.path.join(out_dir, img_dir)
    out_mask_dir = os.path.join(out_dir, mask_dir)

    # create directories for prepared dataset if not already existing
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)

    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    for img_path in image_list:
        # get image name and read image
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        image = io.imread(img_path)

        # based on image name, get path to matching mask and read mask
        mask_path = glob.glob(os.path.join(in_dir, mask_dir, image_name)+"*")[0]
        mask = io.imread(mask_path)

        # divide image into patches
        img_patches = view_as_windows(image,
                                      (patch_width, patch_height, 3),
                                      (3 * patch_width//4, 3 * patch_height//4, 3))
        img_patches = img_patches.reshape(-1, patch_width, patch_height, 3)

        mask_patches = view_as_windows(mask,
                                       (patch_width, patch_height),
                                       (3 * patch_width//4, 3 * patch_height//4))
        mask_patches = mask_patches.reshape(-1, patch_width, patch_height)

        for i in tqdm(range(len(img_patches))):
            # save patches to directory
            target_img_path = os.path.join(out_img_dir, image_name) + "_" + str(i) + ".png"
            target_mask_path = os.path.join(out_mask_dir, image_name) + "_" + str(i) + ".png"

            io.imsave(target_img_path, img_as_ubyte(img_patches[i]))
            io.imsave(target_mask_path, mask_patches[i])


IN_ROOT_DIR = "../dataset"
OUT_ROOT_DIR = "../prepared_dataset"

dataset_name = "Amgad2019"

split_into_patches(os.path.join(IN_ROOT_DIR, dataset_name), os.path.join(OUT_ROOT_DIR, dataset_name))

