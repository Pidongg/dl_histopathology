# Converts the model to torchscript format, to run custom inference on QuPath.
from ultralytics import YOLO

# Load your trained model
model = YOLO("C:/Users/peiya/Desktop/train16/weights/best.pt")

# Export to TorchScript
model.export(format='torchscript')