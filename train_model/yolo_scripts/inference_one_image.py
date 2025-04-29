from ultralytics import YOLO
from PIL import Image


# Load a pretrained YOLO11n model
model = YOLO('C:/Users/peiya/Downloads/train16/weights/best.pt')

# Define path to the image file
# source = "M:/Unused/TauCellDL/images/test/703488/703488 [d=0.98892,x=89715,y=57911,w=507,h=507].png"
source = "M:/Unused/TauCellDL/images/test/703488/703488 [d=0.98892,x=85918,y=55538,w=506,h=506].png"

# Run inference on the source
results = model(source, visualize=False)  # list of Results objects

img_array = results[0].plot()

img = Image.fromarray(img_array[..., ::-1])
img.save('output_0.png')