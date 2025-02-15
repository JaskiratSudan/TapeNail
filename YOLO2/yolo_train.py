# from ultralytics import YOLO

# # Load a YOLOv8 model
# model = YOLO("yolov8n.pt")

# # Train
# model.train(
#     data="dataset/data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     name="yolo_binary_classification"
# )

from ultralytics import YOLO

# Load a pre-trained YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # 'n' is lightweight; use 'm' or 'l' for better accuracy

# Train the model
results = model.train(
    data="data.yaml",  # Roboflow should have generated this
    epochs=5,         # Adjust based on dataset size
    imgsz=640,         # Resize images to 640x640
    batch=16,          # Adjust based on GPU memory
    name="yolo_segmentation",
    project="YOLO_Segmentation",
)