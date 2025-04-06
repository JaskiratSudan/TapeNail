from ultralytics import YOLO

# Load trained YOLO model
<<<<<<< HEAD
model = YOLO("multiclass_det_03-26-25.pt")
=======
model = YOLO("content/YOLOv11_det_640/yolov11_det_640/weights/best.pt")
>>>>>>> parent of 36d8075 (multiclass working and authentication working.)

model.export(format="torchscript")