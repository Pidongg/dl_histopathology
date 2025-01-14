# For YOLOv8
from ultralytics import YOLO

# Load your trained model
model = YOLO('C:/Users/peiya/Downloads/train26/weights/best.pt')

# Export to TorchScript
model.export(format='ONNX')