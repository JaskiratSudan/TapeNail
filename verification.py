#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import time
import os

# ---------------------------
# Config
# ---------------------------
PATTERN_DIR       = "patterns/circle"
MIN_MATCH_COUNT   = 8
RATIO_THRESH      = 0.75
N_FEATURES        = 0       # for SIFT_create
FLANN_INDEX_KDTREE = 1
FLANN_TREES        = 5
FLANN_CHECKS       = 50
# ---------------------------

def root_sift(descriptors):
    if descriptors is None:
        return None
    desc = descriptors.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc)

def extract_color_sift(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    all_kp, all_desc = [], []
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    for ch in (1, 2):  # a, b channels
        kp, desc = sift.detectAndCompute(lab[:, :, ch], None)
        if desc is not None and len(kp) > 0:
            desc = root_sift(desc)
            all_kp.extend(kp)
            all_desc.append(desc)
    if all_desc:
        return all_kp, np.vstack(all_desc)
    return [], None

# Load templates
templates = []
for npz in glob.glob(os.path.join(PATTERN_DIR, "*.npz")):
    data = np.load(npz)
    name = os.path.basename(npz).split(".")[0]
    kp_coords = data["keypoints"]
    desc      = data["descriptors"]
    templates.append({
        "name": name,
        "keypoints": kp_coords,
        "descriptors": desc
    })
    print(f"[Loaded] {name}: {desc.shape[0]} descriptors")

# Setup matcher & camera
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
search_params = dict(checks=FLANN_CHECKS)
flann = cv2.FlannBasedMatcher(index_params, search_params)
cap   = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Denoise + equalize on each Lab channel? (skip here to save time)
    # We just extract color SIFT directly
    kp_frame, desc_frame = extract_color_sift(frame)
    out = frame.copy()

    if desc_frame is not None and len(desc_frame) >= 2:
        for tmpl in templates:
            desc_t = tmpl["descriptors"]
            if desc_t is None or len(desc_t) < 2:
                continue

            matches = flann.knnMatch(desc_t, desc_frame, k=2)
            good = [m for m,n in matches if m.distance < RATIO_THRESH * n.distance]

            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32([
                    tmpl["keypoints"][m.queryIdx] for m in good
                ]).reshape(-1,1,2)
                dst_pts = np.float32([
                    kp_frame[m.trainIdx].pt for m in good
                ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = frame.shape[:2]
                    corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                    transformed = cv2.perspectiveTransform(corners, M)
                    cv2.polylines(out, [np.int32(transformed)], True, (0,255,0), 3)
                    cv2.putText(out, tmpl["name"],
                                tuple(np.int32(transformed[0][0] + [10,-10])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev); prev = now
    cv2.putText(out, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    cv2.imshow("ViKey Verify", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()