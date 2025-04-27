#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time
import psutil
import argparse

# Try to import RPi.GPIO for active light / alarm
try:
    import RPi.GPIO as GPIO
    LED_PIN = 17
    BUZZER_PIN = 27
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    USE_GPIO = True
except ImportError:
    USE_GPIO = False

# --------------------------------
# Configurable Parameters
# --------------------------------
CONTOUR_PARAMS = {
    'min_area': 10,
    'approx_poly_epsilon': 0.2,
    'square_aspect_ratio_min': 0.95,
    'square_aspect_ratio_max': 1.05,
}
IMAGE_PARAMS = {
    'resize_width': 600,
    'blur_kernel_size': (5, 5),
    'canny_threshold1': 65,
    'canny_threshold2': 150,
}
DISPLAY_PARAMS = {
    'contour_color': (0, 255, 0),
    'contour_thickness': 2,
    'text_color': (255, 255, 255),
    'text_thickness': 2,
    'text_scale': 0.5,
}
SIFT_PARAMS   = dict(nfeatures=0, nOctaveLayers=4,
                     contrastThreshold=0.03, edgeThreshold=15, sigma=1.6)
FLANN_PARAMS  = dict(algorithm=1, trees=4)
MATCH_PARAMS  = dict(checks=32)
RATIO_TEST    = 0.2
MIN_MATCHES   = 2
COUNTDOWN     = 5.0
SHAPES        = ["circle", "square", "triangle"]

# --------------------------------
# Helper Functions
# --------------------------------
def denoise(img, h=10):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

def detect_shape(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, CONTOUR_PARAMS['approx_poly_epsilon'] * peri, True)
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        if CONTOUR_PARAMS['square_aspect_ratio_min'] <= ar <= CONTOUR_PARAMS['square_aspect_ratio_max']:
            return "square"
        else:
            return "rectangle"
    else:
        return "circle"

def extract_roi(frame, contour):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    roi = frame[y:y+h, x:x+w]
    return roi, (x, y, w, h)

def draw_unlock(frame, rect):
    x, y, w, h = rect
    cx, cy = x + w//2, y + h//2
    size = min(w, h)//2
    pt1 = (cx - size//2, cy)
    pt2 = (cx, cy + size//2)
    pt3 = (cx + size, cy - size)
    cv2.line(frame, pt1, pt2, (0,255,0), 4)
    cv2.line(frame, pt2, pt3, (0,255,0), 4)
    cv2.putText(frame, "UNLOCKED", (cx-100, cy-size),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

# --------------------------------
# Main
# --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=SHAPES, help="Shape to detect")
    args = parser.parse_args()

    # Prompt for shape if not provided
    shape = args.shape or input(f"Select shape ({', '.join(SHAPES)}): ").strip().lower()
    if shape not in SHAPES:
        print("Invalid shape.")
        exit(1)

    # Prepare SIFT and matcher
    sift = cv2.SIFT_create(**SIFT_PARAMS)
    matcher = cv2.FlannBasedMatcher(FLANN_PARAMS, MATCH_PARAMS)

    # Load templates for selected shape
    templates = []
    templ_dir = os.path.join("patterns", shape)
    if not os.path.isdir(templ_dir):
        print(f"No templates found for '{shape}'.")
        exit(1)

    for fn in os.listdir(templ_dir):
        if fn.lower().endswith(".png"):
            img = cv2.imread(os.path.join(templ_dir, fn), cv2.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(img, None)
            if des is not None and len(des) >= 2:
                templates.append({'kp': kp, 'des': des})
    if not templates:
        print("No valid templates loaded.")
        exit(1)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        exit(1)
    if USE_GPIO:
        GPIO.output(LED_PIN, GPIO.HIGH)

    match_start = None
    unlocked = False

    while True:
        loop_t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing & timing
        t0 = time.time()
        frame = cv2.resize(frame, (IMAGE_PARAMS['resize_width'],
                                   int(frame.shape[0] * IMAGE_PARAMS['resize_width'] / frame.shape[1])))
        proc = denoise(frame)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, IMAGE_PARAMS['blur_kernel_size'], 0)
        edges = cv2.Canny(blurred,
                          IMAGE_PARAMS['canny_threshold1'],
                          IMAGE_PARAMS['canny_threshold2'])
        t1 = time.time()

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kp_frame, des_frame = sift.detectAndCompute(gray, None)
        t2 = time.time()

        # Ensure t_match_start and t3 exist
        t_match_start = t2
        t3 = t2
        match_found = False
        best_rect = (0, 0, 0, 0)

        if des_frame is not None:
            for c in contours:
                if cv2.contourArea(c) < CONTOUR_PARAMS['min_area']:
                    continue
                if detect_shape(c) != shape:
                    continue
                roi, rect = extract_roi(proc, c)
                gray_r = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                kp_r, des_r = sift.detectAndCompute(gray_r, None)
                if des_r is None or len(des_r) < 2:
                    continue
                t_match_start = time.time()
                matches = matcher.knnMatch(des_r, templates[0]['des'], k=2)
                good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
                t3 = time.time()
                if len(good) >= MIN_MATCHES:
                    match_found = True
                    best_rect = rect
                    break

        t_end = time.time()
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        print(f"Pre:{(t1-t0)*1000:.1f}ms FE:{(t2-t1)*1000:.1f}ms "
              f"M:{(t3-t_match_start)*1000:.1f}ms Loop:{(t_end-loop_t0)*1000:.1f}ms "
              f"CPU:{cpu:.1f}% MEM:{mem:.1f}%")

        # Unlock logic
        if match_found:
            x, y, w, h = best_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), DISPLAY_PARAMS['contour_color'],
                          DISPLAY_PARAMS['contour_thickness'])
            if match_start is None:
                match_start = time.time()
            if time.time() - match_start >= COUNTDOWN:
                draw_unlock(frame, best_rect)
                unlocked = True
        else:
            match_start = None

        cv2.imshow("Detection", frame)
        cv2.imshow("Edges", edges)
        if cv2.waitKey(1) & 0xFF == ord('q') or unlocked:
            break

    cap.release()
    cv2.destroyAllWindows()
    if USE_GPIO:
        GPIO.output(LED_PIN, GPIO.LOW)