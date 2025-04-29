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

MATCH_THRESHOLD     = 3
RATIO_THRESH        = 0.70
FLANN_TREES         = 5
FLANN_CHECKS        = 1800
SIFT_MAX_FEATURES   = 1200

LOCK_TIME_SEC       = 1.0    # need 2s of continuous match

# Preproc params
FIXED_SIZE          = (200, 200)
BILATERAL_PARAMS    = (5, 75, 75)
NLM_PARAMS          = {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21}
CLAHE_CLIP          = 2.0
CLAHE_GRID          = (8, 8)

# Contour filtering for circle‐like shapes
MIN_AREA            = 500
CIRCULARITY_THRESH  = 0.7    # 4π·Area/(Perimeter²)

# optimize
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# SIFT & FLANN
sift  = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=FLANN_TREES),
    dict(checks=FLANN_CHECKS)
)

def denoise_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    d, sc, ss = BILATERAL_PARAMS
    s = cv2.bilateralFilter(s, d, sc, ss)
    v = cv2.fastNlMeansDenoising(v, None, **NLM_PARAMS)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

def preprocess(img):
    dn    = denoise_hsv(img)
    cl    = apply_clahe(dn)
    proc  = cv2.resize(cl, FIXED_SIZE)
    gray  = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    return proc, gray

def is_circle(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA: return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0: return False
    circ = 4*math.pi*area/(peri*peri)
    return circ >= CIRCULARITY_THRESH

def load_templates():
    original_map = {}
    templates = []
    for d in sorted(os.listdir(TEMPLATE_DIR)):
        if not d.startswith("pat_"): continue
        pid = d.split("_",1)[1]
        sub = os.path.join(TEMPLATE_DIR, d)
        # grab the first non-crop as the “original”
        for fn in sorted(os.listdir(sub)):
            if fn.endswith(".png") and "_crop" not in fn:
                img = cv2.imread(os.path.join(sub, fn))
                if img is not None:
                    original_map[pid] = img
                    break
        # then load all descriptors
        for fn in sorted(os.listdir(sub)):
            img = cv2.imread(os.path.join(sub, fn))
            if img is None: continue
            _, gray = preprocess(img)
            kp, desc = sift.detectAndCompute(gray, None)
            if desc is None or len(desc)<2: continue
            templates.append({"pid":pid, "kp":kp, "desc":desc})
    if not original_map or not templates:
        raise RuntimeError("No patterns found in " + TEMPLATE_DIR)
    return original_map, templates

def show_pattern_grid(original_map):
    n = len(original_map)
    cols = min(4, n)
    rows = (n + cols -1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols,3*rows))
    axes = axes.flatten() if n>1 else [axes]
    for ax in axes: ax.axis("off")
    for i,(pid,img) in enumerate(sorted(original_map.items())):
        ax = axes[i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Pattern {pid}")
    plt.tight_layout()
    plt.show()

def play_unlock_animation(frame, template_img):
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    # expanding rings
    max_r = int(max(w,h)/1.2)
    for r in np.linspace(0, max_r, 30, dtype=int):
        anim = frame.copy()
        cv2.circle(anim, center, r, (0,200,0), 3)
        alpha = r/max_r
        # fade-in “UNLOCKED”
        cv2.putText(anim, "UNLOCKED", (center[0]-140, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (int(255*alpha),255, int(255*alpha)), 4, cv2.LINE_AA)
        cv2.imshow("Match", anim)
        cv2.imshow("Template", template_img)
        cv2.waitKey(40)
    # final static screen
    final = frame.copy()
    cv2.rectangle(final, (0,0), (w,h), (0,200,0), 10)
    cv2.putText(final, "UNLOCKED", (center[0]-140, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                (0,255,0), 6, cv2.LINE_AA)
    cv2.imshow("Match", final)
    cv2.imshow("Template", template_img)

def main():
    original_map, templates = load_templates()

    while True:
        show_pattern_grid(original_map)
        target = input("Enter pattern ID to unlock (e.g. '01'): ").strip()
        if target not in original_map:
            print("Invalid ID. Try again.")
            continue

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam"); return

        detection_start = None
        hold = 0
        saved_bbox = None
        prev = time.time()
        unlocked = False

        while True:
            ret, frame = cap.read()
            if not ret: break

            # resize
            h0, w0 = frame.shape[:2]
            new_h = int(h0 * FRAME_WIDTH / w0)
            frm = cv2.resize(frame, (FRAME_WIDTH, new_h))
            disp = frm.copy()

            # find circles via contour
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7,7),0)
            edges = cv2.Canny(blur,50,150)
            cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            detected = False
            for c in cnts:
                if not is_circle(c): continue
                x,y,wc,hc = cv2.boundingRect(c)
                roi = frm[y:y+hc, x:x+wc]
                _, gr = preprocess(roi)
                kp2, desc2 = sift.detectAndCompute(gr, None)
                if desc2 is None or len(desc2)<2: continue

                # match only against chosen
                for t in templates:
                    if t["pid"]!=target: continue
                    matches = flann.knnMatch(t["desc"], desc2, k=2)
                    good = [m for m,n in matches if m.distance < RATIO_THRESH*n.distance]
                    if len(good)>=MATCH_THRESHOLD:
                        detected = True
                        saved_bbox = (x,y,wc,hc)
                        break
                if detected: break

            # hold-state
            if detected:
                hold = 5
                if detection_start is None:
                    detection_start = time.time()
                elif time.time()-detection_start >= LOCK_TIME_SEC:
                    unlocked = True
            else:
                hold = max(hold-1,0)
                if hold==0:
                    detection_start = None
                    saved_bbox = None

            # draw box
            if saved_bbox:
                x,y,wc,hc = saved_bbox
                cv2.rectangle(disp,(x,y),(x+wc,y+hc),(0,255,0),2)

            # show FPS
            now = time.time()
            fps = 1.0/(now-prev); prev=now
            cv2.putText(disp,f"FPS:{fps:.1f}",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)

            cv2.imshow("Match", disp)
            cv2.imshow("Template",
                       original_map[target] if not unlocked else original_map[target])

            if unlocked:
                # play the animation once, then block until ANY key
                play_unlock_animation(disp, original_map[target])
                print("🔓 UNLOCKED!  Press any key to continue.")
                cv2.waitKey(0)
                break

            if cv2.waitKey(1)&0xFF==ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()