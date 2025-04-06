from ultralytics import YOLO

# Load trained YOLO model
<<<<<<< HEAD
<<<<<<< HEAD
model = YOLO("multiclass_det_03-26-25.pt")
=======
model = YOLO("content/YOLOv11_det_640/yolov11_det_640/weights/best.pt")
>>>>>>> parent of 36d8075 (multiclass working and authentication working.)
=======
model = YOLO("multiclass_det_03-26-25.pt")
>>>>>>> 4c6a3fceb7123e77680acc2897a8db8ad1f770d6

model.export(format="torchscript")