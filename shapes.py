#!/usr/bin/env python3
import cv2
import imutils
import numpy as np
import os

# --------------------------------
# Configurable Parameters (defaults)
# --------------------------------
CONTOUR_PARAMS = {
    'min_area': 10,
    'approx_poly_epsilon': 0.2,
    'square_aspect_ratio_min': 0.95,
    'square_aspect_ratio_max': 1.05,
}
DISPLAY_PARAMS = {
    'contour_color': (0, 255, 0),
    'contour_thickness': 2,
    'text_color': (255, 255, 255),
    'text_thickness': 2,
    'text_scale': 0.5,
    'status_color': (0, 0, 255),
    'status_scale': 0.8,
}

def get_next_pattern_number(base_dir="patterns"):
    max_num = 0
    for shape in ["circle", "square", "triangle"]:
        d = os.path.join(base_dir, shape)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("pat_") and f.endswith(".png"):
                try:
                    num = int(f.split("_")[1])
                    max_num = max(max_num, num)
                except:
                    pass
    return max_num + 1

def detect_shape(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, CONTOUR_PARAMS['approx_poly_epsilon'] * peri, True)
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        return "square" if CONTOUR_PARAMS['square_aspect_ratio_min'] <= ar <= CONTOUR_PARAMS['square_aspect_ratio_max'] else "rectangle"
    else:
        return "circle"

def extract_shape_region(frame, contour):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    roi = frame[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]
    result = np.zeros((h, w, 4), dtype=np.uint8)
    idx = mask_roi == 255
    result[idx, :3] = roi[idx]
    result[idx, 3] = 255
    return result

def on_change(x):
    pass

def main():
    base_dir = "patterns"
    shapes = ["circle", "square", "triangle"]
    os.makedirs(base_dir, exist_ok=True)
    for s in shapes:
        os.makedirs(os.path.join(base_dir, s), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Create control window
    cv2.namedWindow("Controls")
    # Trackbars: blur kernel size (1–51), canny thresholds (0–255)
    cv2.createTrackbar("Blur k", "Controls", 5, 51, on_change)
    cv2.createTrackbar("Canny Th1", "Controls", 65, 255, on_change)
    cv2.createTrackbar("Canny Th2", "Controls", 150, 255, on_change)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Read trackbar values
        k = cv2.getTrackbarPos("Blur k", "Controls")
        if k < 1: k = 1
        if k % 2 == 0: k += 1
        th1 = cv2.getTrackbarPos("Canny Th1", "Controls")
        th2 = cv2.getTrackbarPos("Canny Th2", "Controls")

        # Resize frame
        frame = imutils.resize(frame, width=600)
        orig = frame.copy()

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(blurred, th1, th2)

        # Find and draw contours
        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            if cv2.contourArea(c) < CONTOUR_PARAMS['min_area']:
                continue
            shape = detect_shape(c)
            if shape in shapes:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                cv2.drawContours(frame, [c], -1, DISPLAY_PARAMS['contour_color'], DISPLAY_PARAMS['contour_thickness'])
                cv2.putText(frame, shape, (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,
                            DISPLAY_PARAMS['text_scale'], DISPLAY_PARAMS['text_color'], DISPLAY_PARAMS['text_thickness'])

        # Status text
        cv2.putText(frame, f"Blur k={k}, Th1={th1}, Th2={th2}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    DISPLAY_PARAMS['status_scale'], DISPLAY_PARAMS['status_color'],
                    DISPLAY_PARAMS['text_thickness'])
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    DISPLAY_PARAMS['status_scale'], DISPLAY_PARAMS['status_color'],
                    DISPLAY_PARAMS['text_thickness'])

        # Show windows
        cv2.imshow("Shape Detection", frame)
        cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            pat = get_next_pattern_number(base_dir)
            print(f"\nCapturing pattern {pat:02d}")
            counters = {s:1 for s in shapes}
            for c in cnts:
                if cv2.contourArea(c) < CONTOUR_PARAMS['min_area']:
                    continue
                shape = detect_shape(c)
                if shape in shapes:
                    roi = extract_shape_region(orig, c)
                    idx = counters[shape]
                    letter = shape[0]
                    fn = os.path.join(base_dir, shape, f"pat_{pat:02d}_{idx:02d}_{letter}.png")
                    cv2.imwrite(fn, roi)
                    print(f"Saved {fn}")
                    counters[shape] += 1
            print("Done.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()