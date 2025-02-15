from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("Best (1).pt")

model.export(format="tflite")