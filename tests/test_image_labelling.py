import unittest
from data_preparation.image_labelling import *

mask_path = "./test_data/test_mask.png"
data_dir = "./test_data"


class TestImageLabelling(unittest.TestCase):
    def test_bboxGeneration(self):
        # generate bboxes from example mask
        bboxes_from_mask, _, _ = bboxes_from_one_mask(mask_path, out_dir=data_dir, yolo=True)
        bboxes_from_mask = bboxes_from_mask.cpu()

        # get path to output label file
        label_path = os.path.join(data_dir, "test_mask.txt")

        # get bboxes from newly created label file
        bboxes_from_labels, _ = bboxes_from_yolo_labels(label_path)
        bboxes_from_labels = bboxes_from_labels.cpu()

        # delete label file
        os.remove(label_path)

        # test results
        torch.testing.assert_close(bboxes_from_mask, bboxes_from_labels)


if __name__ == '__main__':
    unittest.main()
