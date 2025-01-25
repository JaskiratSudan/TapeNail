import cv2
import numpy as np
import time

def create_template_from_webcam(output_path):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None
    
    print("Preparing to capture in 5 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            cap.release()
            return None, None
        
        cv2.imshow('Preparing to Capture', frame)
        cv2.waitKey(1)
    
    # Capture a single frame after 5 seconds
    ret, img = cap.read()
    
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
    
    if not ret:
        print("Failed to capture image from webcam")
        return None, None
        
    # Allow the user to crop the image
    print("Select the region to crop and press Enter or Space. Press Esc to skip cropping.")
    roi = cv2.selectROI('Crop Template', img)
    
    # If the user selects a region
    if roi != (0, 0, 0, 0):  # ROI is (x, y, w, h)
        x, y, w, h = roi
        cropped_img = img[y:y+h, x:x+w]
        cv2.destroyWindow('Crop Template')
    else:
        print("No cropping performed.")
        cropped_img = img
        cv2.destroyWindow('Crop Template')
    
    # Save the cropped template
    cv2.imwrite(output_path, cropped_img)
    
    return cropped_img, img

# Usage
output_path = 'templets/template_from_webcam.png'
template, original_image = create_template_from_webcam(output_path)

# Display the template and original image if they were created
if template is not None and original_image is not None:
    cv2.imshow('Original Captured Image', original_image)
    cv2.imshow('Template', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to create template from webcam image")
