from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("multiclass_det_03-26-25.pt")

model.export(format="tflite")