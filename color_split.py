#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time

# === CONFIGURATION ===
TEMPLATE_DIR       = "patterns/circle"  # folder with circular template images
FRAME_WIDTH        = 640                 # resize width for speed
MIN_RADIUS         = 20                  # ignore tiny circles
MAX_RADIUS         = 200                 # optional upper limit
HOUGH_DP           = 1.2                 # inverse ratio for HoughCircles
HOUGH_MIN_DIST     = 50                  # min distance between circle centers
HOUGH_PARAM1       = 100                 # upper threshold for Canny edge detector
HOUGH_PARAM2       = 30                  # threshold for center detection
MATCH_THRESHOLD    = 14                  # total good matches across BGR
RATIO_THRESH       = 0.90
FLANN_TREES        = 5
FLANN_CHECKS       = 50
SIFT_MAX_FEATURES  = 0                 # limit SIFT keypoints for speed

# Enable OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Initialize SIFT and FLANN
sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=FLANN_TREES),
    dict(checks=FLANN_CHECKS)
)

def denoise_hsv(frame):
    """Denoise frame in HSV: bilateral on S, NLM on V."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.bilateralFilter(s, d=5, sigmaColor=75, sigmaSpace=75)
    v = cv2.fastNlMeansDenoising(v, None, h=10,
                                 templateWindowSize=7,
                                 searchWindowSize=21)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def extract_channel_sift(img):
    """Run SIFT on each BGR channel; return lists of keypoints & descriptors."""
    kps, descs = [], []
    for ch in range(3):
        kp, desc = sift.detectAndCompute(img[:, :, ch], None)
        kps.append(kp)
        descs.append(desc)
    return kps, descs

# --- Load templates: extract per-channel SIFT descriptors and keep the image ---
templates = []
for fn in os.listdir(TEMPLATE_DIR):
    path = os.path.join(TEMPLATE_DIR, fn)
    tpl = cv2.imread(path)
    if tpl is None:
        continue
    kps, descs = extract_channel_sift(tpl)
    templates.append({
        "name": os.path.splitext(fn)[0],
        "image": tpl,
        "kps": kps,
        "descs": descs
    })
    counts = [0 if d is None else len(d) for d in descs]
    print(f"[Loaded] {fn}: B={counts[0]} G={counts[1]} R={counts[2]} descriptors")

if not templates:
    print("No circle templates found in", TEMPLATE_DIR)
    exit(1)

# Prepare display windows
cv2.namedWindow("Circle Match")
cv2.namedWindow("Matched Template")

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit(1)

prev_time = time.time()
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (FRAME_WIDTH,
                               int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
    output = frame.copy()

    # Detect circles via Hough (on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
                               minDist=HOUGH_MIN_DIST,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                               minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    matched_template = None

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        # Pre-denoise full frame once per iteration
        denoised = denoise_hsv(frame)

        for (x, y, r) in circles:
            # Define square ROI around circle
            x1, y1 = max(x - r, 0), max(y - r, 0)
            x2, y2 = min(x + r, frame.shape[1]), min(y + r, frame.shape[0])
            roi = denoised[y1:y2, x1:x2]

            # Extract per-channel SIFT on ROI
            roi_kps, roi_descs = extract_channel_sift(roi)

            # Try each template
            for tmpl in templates:
                total_good = 0
                for ch in range(3):
                    desc_t = tmpl["descs"][ch]
                    desc_r = roi_descs[ch]
                    # Guard: need at least 2 descriptors to match
                    if desc_t is None or desc_r is None or len(desc_t) < 2 or len(desc_r) < 2:
                        continue
                    matches = flann.knnMatch(desc_t, desc_r, k=2)
                    good = [m for m, n in matches if m.distance < RATIO_THRESH * n.distance]
                    total_good += len(good)

                # Check threshold
                if total_good >= MATCH_THRESHOLD:
                    # Draw detection box and label
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(output, tmpl["name"], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    matched_template = tmpl["image"]
                    break  # stop after first match

            if matched_template is not None:
                break  # no need to test other circles

    # Display FPS
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now
    cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Show main window
    cv2.imshow("Circle Match", output)

    # Show matched template, if any
    if matched_template is not None:
        # scale matched template to fixed height
        h = 200
        scale = h / matched_template.shape[0]
        w = int(matched_template.shape[1] * scale)
        disp_tpl = cv2.resize(matched_template, (w, h))
        cv2.imshow("Matched Template", disp_tpl)
    else:
        # clear window if nothing matched
        cv2.imshow("Matched Template", np.zeros((200, 200, 3), dtype=np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()