#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time

# === CONFIGURATION ===
TEMPLATE_DIR       = "patterns/circle"  # root with subdirs pat_## containing .png templates
FRAME_WIDTH        = 640                # resize width for speed
MIN_RADIUS         = 20                 # ignore tiny circles
MAX_RADIUS         = 200                # upper limit for circle radius
HOUGH_DP           = 1.2                # inverse ratio for Hough accumulator
HOUGH_MIN_DIST     = 50                 # min distance between circle centers
HOUGH_PARAM1       = 100                # Canny high threshold
HOUGH_PARAM2       = 30                 # accumulator threshold for circles

MATCH_THRESHOLD    = 10                 # min good matches to accept
RATIO_THRESH       = 0.90               # Lowe’s ratio test
FLANN_TREES        = 5                  # KD-tree count
FLANN_CHECKS       = 250                 # search checks
SIFT_MAX_FEATURES  = 400                # cap SIFT features for speed

FIXED_SIZE         = (200, 200)         # resize templates & ROIs
BILATERAL_PARAMS   = (5, 75, 75)        # d, sigmaColor, sigmaSpace
NLM_PARAMS         = {"h": 10,          # filter strength
                      "templateWindowSize": 7,
                      "searchWindowSize": 21}
CLAHE_CLIP         = 2.0                # contrast limit
CLAHE_GRID         = (8, 8)             # tile grid size

# enable optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# initialize SIFT & FLANN
sift  = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=FLANN_TREES),
    dict(checks=FLANN_CHECKS)
)

def denoise_hsv(bgr):
    """Bilateral on S, NLM on V in HSV space."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    d, sc, ss = BILATERAL_PARAMS
    s = cv2.bilateralFilter(s, d, sc, ss)
    v = cv2.fastNlMeansDenoising(v, None, **NLM_PARAMS)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def apply_clahe(bgr):
    """CLAHE on L channel in Lab space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

def preprocess(img):
    """Denoise, CLAHE, resize, then return both color and gray versions."""
    dn    = denoise_hsv(img)
    cl    = apply_clahe(dn)
    proc  = cv2.resize(cl, FIXED_SIZE)
    gray  = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    return proc, gray

# --- Load templates ---
templates = []
for pat_dir in sorted(os.listdir(TEMPLATE_DIR)):
    sub = os.path.join(TEMPLATE_DIR, pat_dir)
    if not os.path.isdir(sub) or not pat_dir.startswith("pat_"):
        continue
    for fn in sorted(os.listdir(sub)):
        path = os.path.join(sub, fn)
        tpl  = cv2.imread(path)
        if tpl is None:
            continue
        tpl_proc, tpl_gray = preprocess(tpl)
        kp, desc = sift.detectAndCompute(tpl_gray, None)
        if desc is None or len(desc) < 2:
            continue
        templates.append({
            "name": f"{pat_dir}/{os.path.splitext(fn)[0]}",
            "image": tpl_proc,
            "keypoints": kp,
            "descriptors": desc
        })
        print(f"[Loaded] {pat_dir}/{fn}: {len(kp)} keypoints")

if not templates:
    print("No templates found."); exit(1)

cv2.namedWindow("Circle Match")
cv2.namedWindow("Matched Template")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam"); exit(1)

prev = time.time()
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH,
                               int(frame.shape[0]*FRAME_WIDTH/frame.shape[1])))
    output = frame.copy()

    # detect circles
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_full = cv2.medianBlur(gray_full, 5)
    circles   = cv2.HoughCircles(gray_full, cv2.HOUGH_GRADIENT,
                                 dp=HOUGH_DP,
                                 minDist=HOUGH_MIN_DIST,
                                 param1=HOUGH_PARAM1,
                                 param2=HOUGH_PARAM2,
                                 minRadius=MIN_RADIUS,
                                 maxRadius=MAX_RADIUS)

    matched = None

    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            x1, y1 = max(x-r,0), max(y-r,0)
            x2, y2 = min(x+r,frame.shape[1]), min(y+r,frame.shape[0])
            roi = frame[y1:y2, x1:x2]
            proc, gray = preprocess(roi)

            kp2, desc2 = sift.detectAndCompute(gray, None)
            if desc2 is None or len(desc2)<2:
                continue

            # match against each template
            for tmpl in templates:
                matches = flann.knnMatch(tmpl["descriptors"], desc2, k=2)
                good = [m for m,n in matches if m.distance < RATIO_THRESH*n.distance]
                if len(good) >= MATCH_THRESHOLD:
                    cv2.rectangle(output, (x1,y1), (x2,y2), (0,255,0), 3)
                    cv2.putText(output, tmpl["name"], (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    matched = tmpl["image"]
                    break
            if matched is not None:
                break

    # FPS display
    now = time.time()
    fps = 1.0/(now-prev); prev = now
    cv2.putText(output, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    # show results
    cv2.imshow("Circle Match", output)
    if matched is not None:
        cv2.imshow("Matched Template", matched)
    else:
        cv2.imshow("Matched Template",
                   np.zeros((FIXED_SIZE[1],FIXED_SIZE[0],3),np.uint8))

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()