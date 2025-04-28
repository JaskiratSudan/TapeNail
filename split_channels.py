#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt

# === CONFIGURATION ===
TEMPLATE_DIR        = "patterns/circle"
FRAME_WIDTH         = 640

MATCH_THRESHOLD     = 8        # needed per channel
RATIO_THRESH        = 0.90     # Lowe's ratio
FLANN_TREES         = 5
FLANN_CHECKS        = 800
SIFT_MAX_FEATURES   = 1500
HOLD_FRAMES_MAX     = 5
LOCK_TIME_SEC       = 2.0      # seconds to hold detection

# preprocessing params
FIXED_SIZE          = (200, 200)
BILATERAL_PARAMS    = (5, 75, 75)
NLM_PARAMS          = {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21}
CLAHE_CLIP          = 2.0
CLAHE_GRID          = (8, 8)

# contour filtering
MIN_AREA            = 500
CIRCULARITY_THRESH  = 0.7      # 4π·Area/(Perimeter²)

# initialize SIFT & FLANN
sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=FLANN_TREES), dict(checks=FLANN_CHECKS)
)

# preprocessing pipeline
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
    return cv2.resize(enhanced, FIXED_SIZE)

# test for circularity
def is_circle(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA: return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0: return False
    circ = 4*math.pi*area/(peri*peri)
    return circ >= CIRCULARITY_THRESH

# display one original per pattern
def show_pattern_grid(original_map):
    n = len(original_map)
    cols = min(4, n)
    rows = (n + cols - 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if n>1 else [axes]
    for ax in axes: ax.axis('off')
    for i, (pid, img) in enumerate(sorted(original_map.items())):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Pattern {pid}")
    plt.tight_layout()
    plt.show()

# smooth unlock animation: expanding rings + pulsing text
def play_unlock_animation(frame, template):
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    max_r = int(min(w, h) * 0.6)
    base = frame.copy()
    for step in range(30):
        overlay = base.copy()
        r = int((step/29) * max_r)
        alpha = 1.0 - (step/29)
        cv2.circle(overlay, center, r, (0, 255, 0), 4)
        # pulsing scale
        scale = 1.0 + 0.3 * math.sin(math.pi * step/29)
        cv2.putText(overlay, "UNLOCKED", (center[0]-150, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), 4, cv2.LINE_AA)
        blended = cv2.addWeighted(overlay, alpha, base, 1-alpha, 0)
        cv2.imshow('Match', blended)
        cv2.imshow('Template', template)
        cv2.waitKey(50)
    # final static display
    final = base.copy()
    cv2.rectangle(final, (0,0), (w,h), (0,255,0), 12)
    cv2.putText(final, "UNLOCKED", (center[0]-150, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 6, cv2.LINE_AA)
    cv2.imshow('Match', final)
    cv2.imshow('Template', template)
    cv2.waitKey(0)

# load templates
templates = []
original_map = {}
for d in sorted(os.listdir(TEMPLATE_DIR)):
    sub = os.path.join(TEMPLATE_DIR, d)
    if not os.path.isdir(sub) or not d.startswith('pat_'): continue
    pid = d.split('_',1)[1]
    # load representative original
    for fn in sorted(os.listdir(sub)):
        if fn.endswith('.png') and '_crop' not in fn:
            img = cv2.imread(os.path.join(sub, fn))
            if img is not None:
                original_map[pid] = preprocess(img)
                break
    # load channel descriptors
    for fn in sorted(os.listdir(sub)):
        img = cv2.imread(os.path.join(sub, fn))
        if img is None: continue
        proc = preprocess(img)
        chans = cv2.split(proc)
        descs = []
        ok = True
        for ch in chans:
            kp, desc = sift.detectAndCompute(ch, None)
            if desc is None or len(desc) < MATCH_THRESHOLD:
                ok = False; break
            descs.append(desc)
        if ok:
            templates.append({'pid': pid, 'descs': descs})

cv2.namedWindow('Match')
cv2.namedWindow('Template')

while True:
    show_pattern_grid(original_map)
    target = input("Enter pattern ID to unlock: ").strip()
    if target not in original_map:
        print("Invalid ID, try again.")
        continue

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        break

    prev = time.time()
    hold = 0
    saved_box = None
    start_time = None
    unlocked = False

    print("Press 'r' to restart, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        h0, w0 = frame.shape[:2]
        new_h = int(h0 * FRAME_WIDTH / w0)
        frm = cv2.resize(frame, (FRAME_WIDTH, new_h))
        disp = frm.copy()

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        temp_box = None

        for c in cnts:
            if not is_circle(c): continue
            x,y,wc,hc = cv2.boundingRect(c)
            roi = frm[y:y+hc, x:x+wc]
            proc = preprocess(roi)
            chans = cv2.split(proc)
            valid = True
            for i,ch in enumerate(chans):
                kp2, desc2 = sift.detectAndCompute(ch, None)
                if desc2 is None:
                    valid = False; break
                # channel match threshold check
                goods = [m for m,n in flann.knnMatch([t for t in templates if t['pid']==target][0]['descs'][i], desc2, k=2)
                         if m.distance < RATIO_THRESH*n.distance]
                if len(goods) < MATCH_THRESHOLD:
                    valid = False; break
            if valid:
                detected = True
                temp_box = (x,y,wc,hc)
                break

        if detected:
            hold = HOLD_FRAMES_MAX
            saved_box = temp_box
            if start_time is None:
                start_time = time.time()
        else:
            hold = max(hold-1, 0)
            if hold == 0:
                saved_box = None
                start_time = None

        # unlock
        if not unlocked and start_time and time.time()-start_time >= LOCK_TIME_SEC:
            unlocked = True
            play_unlock_animation(disp, original_map[target])

        # draw box and label
        if saved_box:
            x,y,wc,hc = saved_box
            cv2.rectangle(disp, (x,y), (x+wc,y+hc), (0,255,0), 2)
            cv2.putText(disp, f"Pattern {target}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        now = time.time()
        fps = 1.0/(now-prev); prev = now
        cv2.putText(disp, f"FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.imshow('Match', disp)
        cv2.imshow('Template', np.zeros_like(original_map[target]) if not unlocked else original_map[target])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            break
        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); exit(0)

    cap.release()
    cv2.destroyAllWindows()
