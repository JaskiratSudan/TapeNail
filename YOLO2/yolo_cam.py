from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load model
model = YOLO("Gtapenail2/weights/best.pt")

# Object classes
classNames = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4",
              "Pattern 5", "Pattern 6", "Pattern 7"]

# Assign a distinct BGR color per class
COLORS = [
    (255, 0, 255),  # magenta
    (0, 255, 0),    # green
    (0, 255, 255),  # yellow
    (255, 0, 0),    # blue
    (0, 165, 255),  # orange
    (255, 255, 0),  # cyan
    (128, 0, 128),  # purple
]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    # Prepare cutout display offsets
    y_offset = 10
    cutout_size = 100
    x_cutout_pos = img.shape[1] - cutout_size - 10

    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0].item())
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Class index
            cls = int(box.cls[0].item())
            label = f"{classNames[cls]} {confidence:.2f}"

            # Pick color for this class
            color = COLORS[cls % len(COLORS)]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Compute text size & position
            (tw, th), baseline = cv2.getTextSize(label,
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 0.6, 2)
            text_x = x1
            text_y = max(y1 - 10, th + 10)

            # Draw filled rectangle behind text for readability
            cv2.rectangle(img,
                          (text_x, text_y - th - baseline),
                          (text_x + tw, text_y + baseline),
                          color, cv2.FILLED)

            # Put class name + confidence above box
            cv2.putText(img, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            # Extract and show cutout
            cutout = img[y1:y2, x1:x2]
            if cutout.size:
                cut = cv2.resize(cutout, (cutout_size, cutout_size))
                img[y_offset:y_offset + cutout_size,
                    x_cutout_pos:x_cutout_pos + cutout_size] = cut
                cv2.rectangle(img,
                              (x_cutout_pos, y_offset),
                              (x_cutout_pos + cutout_size,
                               y_offset + cutout_size),
                              (0, 255, 0), 2)
                y_offset += cutout_size + 10

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
