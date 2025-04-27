#!/usr/bin/env python3
import cv2
import imutils
import numpy as np
import os

# --------------------------------
# Configurable Parameters
# --------------------------------
BASE_DIR           = "patterns"
SHAPE              = "circle"
MIN_AREA           = 500
APPROX_EPSILON     = 0.02
CANNNY_THRESH      = (50, 150)
SIFT_MAX_FEATURES  = 0    # 0 = default (unlimited)
BILATERAL_PARAMS   = (5, 75, 75)  # used in RootSIFT normalization only
# --------------------------------

def get_next_pattern_number():
    folder = os.path.join(BASE_DIR, SHAPE)
    existing = [f for f in os.listdir(folder) if f.startswith("pat_") and f.endswith(".png")]
    nums = [int(f.split("_")[1]) for f in existing]
    return max(nums, default=0) + 1

def detect_shape(contour):
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, APPROX_EPSILON * peri, True)
    return "circle" if len(approx) > 4 else None

def extract_shape_region(frame, contour):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    roi = frame[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = roi
    rgba[..., 3]  = mask_roi
    return rgba

def root_sift(descriptors):
    # L1-normalize + Hellinger (sqrt) transform
    if descriptors is None:
        return None
    desc = descriptors.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc)

def extract_color_sift(img_bgr):
    """
    Convert BGR → Lab, extract SIFT on a & b channels, RootSIFT, and concatenate.
    Returns (keypoints, descriptors).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    all_kp, all_desc = [], []
    sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    for ch in (1, 2):  # a, b channels
        single = lab[:, :, ch]
        kp, desc = sift.detectAndCompute(single, None)
        if desc is not None and len(kp) > 0:
            desc = root_sift(desc)
            all_kp.extend(kp)
            all_desc.append(desc)
    if all_desc:
        return all_kp, np.vstack(all_desc)
    return [], None

def main():
    os.makedirs(os.path.join(BASE_DIR, SHAPE), exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam"); return

    print("→ Press 'c' to capture circular patterns, 'q' to quit.")
    pat_num = get_next_pattern_number()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_disp = imutils.resize(frame, width=600)
        orig      = frame_disp.copy()
        gray      = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY)
        edges     = cv2.Canny(gray, *CANNNY_THRESH)

        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA: continue
            if detect_shape(c) == SHAPE:
                cv2.drawContours(frame_disp, [c], -1, (0,255,0), 2)

        cv2.imshow("Registration View", frame_disp)
        cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print(f"\n[+] Capturing pattern #{pat_num:02d}")
            count = 0
            for c in cnts:
                if cv2.contourArea(c) < MIN_AREA: continue
                if detect_shape(c) != SHAPE: continue

                roi_rgba = extract_shape_region(orig, c)
                roi_bgr  = cv2.cvtColor(roi_rgba, cv2.COLOR_BGRA2BGR)
                kps, desc = extract_color_sift(roi_bgr)
                if desc is None or len(kps) == 0:
                    print("   [!] no color features; skipping ROI")
                    continue

                # save image
                img_fn = os.path.join(BASE_DIR, SHAPE,
                    f"pat_{pat_num:02d}_{count:02d}_{SHAPE[0]}.png")
                cv2.imwrite(img_fn, roi_rgba)

                # save features
                kp_coords = np.array([kp.pt for kp in kps], dtype=np.float32)
                feat_fn = os.path.join(BASE_DIR, SHAPE,
                    f"pat_{pat_num:02d}_{count:02d}_{SHAPE[0]}.npz")
                np.savez(feat_fn, keypoints=kp_coords, descriptors=desc)

                print(f"   • ROI → {img_fn}")
                print(f"   • features → {feat_fn} ({len(kps)} kps)")
                count += 1

            if count == 0:
                print("   [!] no circles captured.")
            else:
                pat_num += 1
                print("Done.\n")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()