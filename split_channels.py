#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt

# === CONFIGURATION ===
TEMPLATE_DIR      = "patterns/circle"
FRAME_WIDTH       = 960
MATCH_THRESHOLD   = 5       # matches needed per template
RATIO_THRESH      = 0.85    # Lowe's ratio test
FLANN_TREES       = 5
FLANN_CHECKS      = 1200
SIFT_MAX_FEATURES = 2200
HOLD_FRAMES_MAX   = 5       # frames to hold detection state
UNLOCK_DELAY      = 1.0     # seconds for sustained correct detection

# Preprocessing parameters
FIXED_SIZE        = (200, 200)
BILATERAL_PARAMS  = (5, 75, 75)
NLM_PARAMS        = {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21}
CLAHE_CLIP        = 2.0
CLAHE_GRID        = (8, 8)

# Contour filtering
MIN_AREA           = 500    # minimum area of a detected contour
CIRCULARITY_THRESH = 0.7    # 4π·Area/(Perimeter²)

# Initialize SIFT & FLANN matcher
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=FLANN_TREES), dict(checks=FLANN_CHECKS)
)

# --- Helper Functions ---

def preprocess(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.bilateralFilter(s, *BILATERAL_PARAMS)
    v = cv2.fastNlMeansDenoising(v, None, **NLM_PARAMS)
    den = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)
    proc = cv2.resize(enhanced, FIXED_SIZE)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    return proc, gray


def is_circle(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False
    circ = 4 * math.pi * area / (peri * peri)
    return circ >= CIRCULARITY_THRESH


def show_pattern_grid(original_map):
    keys = sorted(original_map.keys())
    n = len(keys)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if n>1 else [axes]
    for ax in axes:
        ax.axis('off')
    for i, pid in enumerate(keys):
        axes[i].imshow(cv2.cvtColor(original_map[pid], cv2.COLOR_BGR2RGB))
        axes[i].set_title(pid)
    plt.tight_layout()
    plt.show()


def detect_pattern(desc2):
    for tmpl in templates:
        matches = flann.knnMatch(tmpl['desc'], desc2, k=2)
        good = [m for m, n in matches if m.distance < RATIO_THRESH * n.distance]
        if len(good) >= MATCH_THRESHOLD:
            return tmpl['pattern_id']
    return None


def play_unlock_animation(frame, template):
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    base = frame.copy()
    for i in range(20):
        alpha = i/19.0
        overlay = base.copy()
        cv2.putText(overlay, 'UNLOCKED', (center[0]-140, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
        blended = cv2.addWeighted(overlay, alpha, base, 1-alpha, 0)
        cv2.imshow('Match', blended)
        cv2.imshow('Template', template)
        cv2.waitKey(50)
    final = base.copy()
    cv2.rectangle(final, (0,0), (w,h), (0,255,0), 8)
    cv2.putText(final, 'UNLOCKED', (center[0]-140, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,255,0), 6, cv2.LINE_AA)
    cv2.imshow('Match', final)
    cv2.imshow('Template', template)
    cv2.waitKey(0)

# --- Load Templates ---
original_map = {}
templates = []
for d in sorted(os.listdir(TEMPLATE_DIR)):
    sub = os.path.join(TEMPLATE_DIR, d)
    if not os.path.isdir(sub) or not d.startswith('pat_'):
        continue
    pid = d.split('_',1)[1]
    for fn in sorted(os.listdir(sub)):
        if fn.endswith('.png') and '_crop' not in fn:
            img = cv2.imread(os.path.join(sub, fn))
            if img is not None:
                original_map[pid] = preprocess(img)[0]
                break
    for fn in sorted(os.listdir(sub)):
        img = cv2.imread(os.path.join(sub, fn))
        if img is None:
            continue
        _, gray = preprocess(img)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(desc) < MATCH_THRESHOLD:
            continue
        templates.append({'pattern_id': pid, 'desc': desc})

if not templates:
    print('No templates found.')
    exit(1)

cv2.namedWindow('Match')
cv2.namedWindow('Template')

# --- Main Loop ---
while True:
    show_pattern_grid(original_map)
    selected = input('Pattern ID to unlock: ').strip()
    if selected not in original_map:
        print('Invalid ID.')
        continue

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        break

    prev = time.time()
    hold = 0
    saved_box = None
    detection_start = None

    print("Press 'r' to restart, 'q' to quit.")
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        h0, w0 = frame.shape[:2]
        new_h = int(h0 * FRAME_WIDTH / w0)
        frm = cv2.resize(frame, (FRAME_WIDTH, new_h))
        disp = frm.copy()

        # Contour detection timing
        t0 = time.time()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_contours = (time.time() - t0) * 1000

        detections = []
        t_match_total = 0
        # Matching timing
        for c in cnts:
            if not is_circle(c):
                continue
            x,y,wc,hc = cv2.boundingRect(c)
            roi = frm[y:y+hc, x:x+wc]
            _, gray_roi = preprocess(roi)
            kp2, desc2 = sift.detectAndCompute(gray_roi, None)
            if desc2 is None:
                continue
            t1 = time.time()
            pid = detect_pattern(desc2)
            t_match_total += (time.time() - t1) * 1000
            if pid:
                detections.append((x,y,wc,hc,pid))

        # Handle correct detection timer
        correct_seen = any(pid==selected for *_, pid in detections)
        if correct_seen:
            if detection_start is None:
                detection_start = time.time()
            hold = HOLD_FRAMES_MAX
            saved_box = next((box for box in detections if box[4]==selected), None)
        else:
            hold = max(hold-1, 0)
            if hold == 0:
                detection_start = None
                saved_box = None

        # Unlock condition
        if detection_start and time.time() - detection_start >= UNLOCK_DELAY:
            play_unlock_animation(disp, original_map[selected])
            break

        # Draw boxes
        for x,y,wc,hc,pid in detections:
            color = (0,255,0) if pid==selected else (0,0,255)
            cv2.rectangle(disp, (x,y), (x+wc,y+hc), color, 2)
            cv2.putText(disp, pid, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Latency overlay
        total_latency = (time.time() - loop_start) * 1000
        cv2.putText(disp, f"Contour:{t_contours:.1f}ms Match:{t_match_total:.1f}ms Total:{total_latency:.1f}ms",
                    (10, new_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # FPS
        now = time.time()
        fps = 1.0/(now-prev); prev = now
        cv2.putText(disp, f"FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # Display
        cv2.imshow('Match', disp)
        cv2.imshow('Template', original_map[selected] if detections else np.zeros_like(original_map[selected]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            break
        elif key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); exit(0)

    cap.release()
    cv2.destroyAllWindows()