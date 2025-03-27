import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation.image_labelling import bboxes_from_yolo_labels

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from pdq_utils import compute_segmentation_mask_for_bbox, visualize_segmentation

bbox, labels = bboxes_from_yolo_labels('C:/Users/peiya/Downloads/703488 [d=0.98892,x=39399,y=22785,w=506,h=506].txt')
for cur_bbox in bbox:
    x0, y0, x1, y1 = cur_bbox
    image_path = 'C:/Users/peiya/Downloads/703488 [d=0.98892,x=39399,y=22785,w=506,h=506].png'
    segmentation_mask_path = 'C:/Users/peiya/Downloads/703488 [d=0.98892,x=39399,y=22785,w=506,h=506]-labelled.png'

    segmentation_mask = compute_segmentation_mask_for_bbox(x0, y0, x1, y1, segmentation_mask_path)

    # Visualize the result
    visualize_segmentation(
        image_path=image_path,
        segmentation_mask=segmentation_mask,
        bbox=[x0, y0, x1, y1],
        save_path='segmentation_visualization.png'
    )