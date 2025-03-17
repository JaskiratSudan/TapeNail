from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("multi_class det_yolov11_256/weights/best.pt")

model.export(format="tflite")