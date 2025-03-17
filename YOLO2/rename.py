import os
import shutil

def copy_and_rename_files(src_dir, dst_dir, base_name='file'):
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        print(f"Source directory does not exist: {src_dir}")
        return

    # Ensure the destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"Created destination directory: {dst_dir}")

    # Define subdirectories to process
    subdirs = ['train', 'valid', 'test']

    for subdir in subdirs:
        src_images_dir = os.path.join(src_dir, subdir, 'images')
        src_labels_dir = os.path.join(src_dir, subdir, 'labels')
        dst_images_dir = os.path.join(dst_dir, subdir, 'images')
        dst_labels_dir = os.path.join(dst_dir, subdir, 'labels')

        # Create destination directories if they don't exist
        os.makedirs(dst_images_dir, exist_ok=True)
        os.makedirs(dst_labels_dir, exist_ok=True)

        # Get list of image and label files
        images = sorted([f for f in os.listdir(src_images_dir) if f.endswith('.jpg')])
        labels = sorted([f for f in os.listdir(src_labels_dir) if f.endswith('.txt')])

        # Check if the number of images and labels match
        if len(images) != len(labels):
            print(f"Mismatch in number of images and labels in {subdir}. Skipping this directory.")
            continue

        # Copy and rename files
        for i, (img, lbl) in enumerate(zip(images, labels)):
            # Generate new names
            new_img_name = f"{base_name}_{i+1:03d}.jpg"
            new_lbl_name = f"{base_name}_{i+1:03d}.txt"

            # Full paths for source and destination
            src_img_path = os.path.join(src_images_dir, img)
            src_lbl_path = os.path.join(src_labels_dir, lbl)
            dst_img_path = os.path.join(dst_images_dir, new_img_name)
            dst_lbl_path = os.path.join(dst_labels_dir, new_lbl_name)

            # Check if source files exist
            if not os.path.exists(src_img_path):
                print(f"Image file not found: {src_img_path}. Skipping.")
                continue
            if not os.path.exists(src_lbl_path):
                print(f"Label file not found: {src_lbl_path}. Skipping.")
                continue

            # Copy and rename
            shutil.copy2(src_img_path, dst_img_path)
            shutil.copy2(src_lbl_path, dst_lbl_path)

            print(f"Copied and renamed: {src_img_path} -> {dst_img_path}")
            print(f"Copied and renamed: {src_lbl_path} -> {dst_lbl_path}")

# Example usage
source_directory = 'dataset'  # Replace with your source directory
destination_directory = 'dataset01'  # Replace with your destination directory
copy_and_rename_files(source_directory, destination_directory, base_name='image')