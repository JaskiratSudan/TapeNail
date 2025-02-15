import os
import glob
import random
import shutil

# Define paths
dataset_path = "dataset"
output_path = "yolo_dataset"
os.makedirs(output_path, exist_ok=True)

# Create YOLO dataset structure
for folder in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

# Get all images
image_files = glob.glob(os.path.join(dataset_path, "*.jpg")) + glob.glob(os.path.join(dataset_path, "*.png"))

# Shuffle dataset
random.shuffle(image_files)

# Split dataset (70% train, 20% val, 10% test)
train_split = int(0.7 * len(image_files))
val_split = int(0.9 * len(image_files))  # Remaining 10% goes to test

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# Function to create YOLO labels (full-image bounding box)
def create_yolo_label(image_path, label_folder, class_id=0):
    filename = os.path.basename(image_path).split('.')[0] + ".txt"
    label_path = os.path.join(label_folder, filename)
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # Full-image bounding box

# Copy files and create labels
for files, img_folder, lbl_folder in [(train_files, "images/train", "labels/train"),
                                      (val_files, "images/val", "labels/val"),
                                      (test_files, "images/test", "labels/test")]:
    for img in files:
        shutil.copy(img, os.path.join(output_path, img_folder))
        create_yolo_label(img, os.path.join(output_path, lbl_folder))

# Create dataset.yaml file
dataset_yaml = f"""train: {os.path.abspath(output_path)}/images/train
val: {os.path.abspath(output_path)}/images/val
test: {os.path.abspath(output_path)}/images/test

nc: 1
names: ["pattern"]
"""

with open(os.path.join(output_path, "dataset.yaml"), "w") as f:
    f.write(dataset_yaml)

print("Dataset preparation complete. Ready for YOLO training!")