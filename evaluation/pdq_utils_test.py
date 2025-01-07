from pdq_utils import compute_segmentation_mask_for_bbox
x0, y0, w, h = [0.1932938856015779, 0.1383399209486166, 0.1893491124260355, 0.1422924901185771]

x1 = x0 + w
y1 = y0 + h

print(x1, y1)
x0 *= 512
y0 *= 512
x1 *= 512
y1 *= 512

segmentation_mask_path = 'M:/Unused/TauCellDL/test_images_seg/747331kept/747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=65506,y=15190,w=507,h=506]-labelled.png'

segmentation_mask = compute_segmentation_mask_for_bbox(x0, y0, x1, y1, segmentation_mask_path)

print(segmentation_mask)