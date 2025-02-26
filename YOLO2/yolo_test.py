import os
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("content/YOLO_Segmentation/yolo_segmentation/weights/best.pt")

# Specify the paths for the 'test' folder and 'output' folder
test_folder = "dataset/test/images"
output_folder = "output_imgs"
os.makedirs(output_folder, exist_ok=True)

# Get the list of images in the 'test' folder
image_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Run batched inference on the images
results = model(image_files, stream=True)
# print("Results Shape --->", len(results))

# Process results generator
for i, result in enumerate(results):
    # Extract outputs (boxes, masks, keypoints, etc.)
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    
    # Display the result
    result.show()

    # Save the result to the output folder with a unique filename
    output_path = os.path.join(output_folder, f"result_{i}.jpg")
    result.save(filename=output_path)

print("Inference completed and results saved in the 'output' folder.")