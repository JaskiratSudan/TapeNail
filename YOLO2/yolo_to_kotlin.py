from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("content/YOLOv11_det_640/yolov11_det_640/weights/best.pt")

model.export(format="torchscript")