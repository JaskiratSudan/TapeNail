import cv2
import os
import numpy as np
import time

# === CONFIGURATION ===
TEMPLATE_DIR       = "patterns/circle"
MIN_MATCH_COUNT    = 8
FLANN_INDEX_KDTREE = 1
RATIO_THRESH       = 0.75

# === SIFT + FLANN SETUP ===
sift = cv2.SIFT_create(nfeatures=300)
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def root_sift(descriptors):
    if descriptors is None:
        return None
    desc = descriptors.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc)

def extract_ab_sift(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    all_kp, all_desc = [], []
    for ch in (1, 2):  # a, b channels
        kp, desc = sift.detectAndCompute(lab[:, :, ch], None)
        if desc is not None:
            desc = root_sift(desc)
            all_kp.extend(kp)
            all_desc.append(desc)
    if all_desc:
        return all_kp, np.vstack(all_desc)
    else:
        return [], None

# === LOAD TEMPLATES ===
templates = []
template_kp_desc = []
template_names = []

for fn in sorted(os.listdir(TEMPLATE_DIR)):
    path = os.path.join(TEMPLATE_DIR, fn)
    img  = cv2.imread(path)
    if img is None:
        continue
    kp, desc = extract_ab_sift(img)
    templates.append(img)
    template_kp_desc.append((kp, desc))
    template_names.append(fn)
    print(f"[INFO] Loaded '{fn}' → {len(kp)} keypoints")

# === START VIDEO ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Press 'q' to quit.")
while True:
    t0, ret = time.time(), cap.read()
    if not ret[0]:
        break
    frame = ret[1]
    kp_frame, desc_frame = extract_ab_sift(frame)
    output = frame.copy()

    # --- GUARD: need at least 2 frame descriptors to do k=2 matching ---
    if desc_frame is not None and len(desc_frame) >= 2:
        for idx, (tmpl, (kp_t, desc_t)) in enumerate(zip(templates, template_kp_desc)):
            # --- GUARD: skip templates with <2 descriptors ---
            if desc_t is None or len(desc_t) < 2:
                continue

            matches = flann.knnMatch(desc_t, desc_frame, k=2)
            good = [m for m, n in matches if m.distance < RATIO_THRESH * n.distance]

            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = tmpl.shape[:2]
                    quad = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
                    dst  = cv2.perspectiveTransform(quad, M)
                    cv2.polylines(output, [np.int32(dst)], True, (0,255,0), 2)
                    cv2.putText(output, template_names[idx],
                                tuple(np.int32(dst[0][0])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # draw FPS
    fps = 1.0 / (time.time() - t0)
    cv2.putText(output, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Reflection-Robust Color SIFT", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()