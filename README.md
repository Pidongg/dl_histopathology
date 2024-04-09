# dl_histopathology

Run prepare_data to get data into the format needed for training. (Performs patching and extracts bounding boxes from mask information.)

Datasets should be stored in a directory named for that dataset, with subdirectories "images" and "masks".

Masks are stored as single-channel .png images, with each pixel value indicating that pixel's class.