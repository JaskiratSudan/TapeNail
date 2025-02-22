from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load model
model = YOLO("Best (1).pt")

# Object classes
classNames = ["Pattern"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Set the minimum confidence score

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    print("Results Shape --->", results.shape())

    # Coordinates
    y_offset = 10  # Offset for displaying cutouts
    cutout_size = 100  # Size of extracted object display
    x_cutout_pos = img.shape[1] - cutout_size - 10  # Position to place cutouts on the right side

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Confidence
            confidence = round(box.conf[0].item(), 2)

            # Only process if confidence is above threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                print("Confidence --->", confidence)

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int values

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Display class and confidence on bounding box
                label = f"{classNames[cls]} {confidence}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Extract object cutout
                cutout = img[y1:y2, x1:x2]
                if cutout.size != 0:
                    cutout = cv2.resize(cutout, (cutout_size, cutout_size))  # Resize cutout
                    img[y_offset:y_offset + cutout_size, x_cutout_pos:x_cutout_pos + cutout_size] = cutout  # Place cutout on right side

                    # Draw border around cutout only
                    cv2.rectangle(img, (x_cutout_pos, y_offset), (x_cutout_pos + cutout_size, y_offset + cutout_size),
                                  (0, 255, 0), 2)

                    # Adjust offset for next object
                    y_offset += cutout_size + 10

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()