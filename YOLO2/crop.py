import os
import cv2
import numpy as np

def yolo_to_polygon(yolo_coords, img_width, img_height):
    """
    Convert YOLO format coordinates to polygon coordinates.
    """
    coords = np.array(yolo_coords, dtype=float)
    polygon = coords.reshape(-1, 2)
    polygon[:, 0] *= img_width  # Scale x-coordinates
    polygon[:, 1] *= img_height  # Scale y-coordinates
    return polygon.astype(int)

def crop_and_save_mask(image_path, mask_coords, output_path):
    """
    Crop the image based on the mask coordinates and save the cropped image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        return

    img_height, img_width = image.shape[:2]
    
    # Convert YOLO coordinates to polygon
    polygon = yolo_to_polygon(mask_coords, img_width, img_height)
    
    # Create a mask from the polygon
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    
    # Bitwise AND to extract the region of interest
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(polygon)
    
    # Ensure the bounding box is valid
    if w == 0 or h == 0:
        print(f"Warning: Invalid bounding box for {image_path}. Skipping.")
        return
    
    # Crop the image using the bounding box
    cropped_image = masked_image[y:y+h, x:x+w]
    
    # Save the cropped image
    if cropped_image.size == 0:
        print(f"Warning: Cropped image is empty for {image_path}. Skipping.")
    else:
        cv2.imwrite(output_path, cropped_image)

def process_folder(root_folder, output_folder):
    """
    Process all images and masks in the dataset folder.
    """
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(root_folder, split, 'images')
        labels_dir = os.path.join(root_folder, split, 'labels')
        output_split_dir = os.path.join(output_folder, split)
        
        os.makedirs(output_split_dir, exist_ok=True)
        
        for image_name in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))
            
            if not os.path.exists(label_path):
                print(f"Warning: No label file found for {image_name}. Skipping.")
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    try:
                        mask_coords = list(map(float, line.strip().split()[1:]))
                        output_image_path = os.path.join(output_split_dir, f"{image_name.replace('.jpg', f'_{i}.jpg')}")
                        crop_and_save_mask(image_path, mask_coords, output_image_path)
                    except Exception as e:
                        print(f"Error processing {image_name}: {e}")

# Define the root folder and output folder
root_folder = 'dataset'
output_folder = 'dataset/cropped'

# Process the dataset
process_folder(root_folder, output_folder)