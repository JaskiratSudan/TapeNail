from ultralytics import YOLO

# Load trained YOLO model
# model = YOLO("multiclass_det_03-26-25.pt")
# model = YOLO("content/YOLOv11_det_640/yolov11_det_640/weights/best.pt")
model = YOLO("tapenail2/weights/best.pt")

model.export(format="tflite")