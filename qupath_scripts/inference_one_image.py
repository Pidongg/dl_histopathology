from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO('C:/Users/peiya/Downloads/train26/weights/best.torchscript')

# Define path to the image file
source = "M:/Unused/TauCellDL/images/test/703488/703488 [d=0.98892,x=103481,y=45570,w=506,h=506].png"

# Run inference on the source
results = model(source)  # list of Results objects

print(results[0].boxes)