#!/usr/bin/env python3
"""
ViKey reader  ▪  with frame-level logging for experiments
---------------------------------------------------------
New features (all marked  # === LOG === ):
  • Per-frame CSV log of:
        frame_idx, tp, fp, contour_ms, match_ms, total_ms, fps
  • Aggregated run summary (TP, FN, FP, accuracy, mean_latency, std_latency)
  • Auto-named log file: logs/read_<PATTERNID>_<YYYYMMDD-HHMMSS>.csv
No other behaviour or parameters changed.
"""
import cv2, numpy as np, os, time, math, csv, datetime, statistics
import matplotlib.pyplot as plt
# --------------[ original imports / config UNCHANGED ]-----------------
TEMPLATE_DIR="patterns/circle"; FRAME_WIDTH=960; MATCH_THRESHOLD=5
RATIO_THRESH=0.85; FLANN_TREES=5; FLANN_CHECKS=1200; SIFT_MAX_FEATURES=2200
HOLD_FRAMES_MAX=5; UNLOCK_DELAY=1.0
FIXED_SIZE=(200,200); BILATERAL_PARAMS=(5,75,75)
NLM_PARAMS={"h":10,"templateWindowSize":7,"searchWindowSize":21}
CLAHE_CLIP=2.0; CLAHE_GRID=(8,8)
MIN_AREA=500; CIRCULARITY_THRESH=0.7
cv2.setUseOptimized(True); cv2.setNumThreads(4)
sift=cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
flann=cv2.FlannBasedMatcher(dict(algorithm=1,trees=FLANN_TREES),
                            dict(checks=FLANN_CHECKS))
# ------------------------[ helper functions UNCHANGED ]----------------
# ...  preprocess(), is_circle(), show_pattern_grid(), detect_pattern(),
#      play_unlock_animation()  (identical to your original script) ...
# ----------------------------------------------------------------------

# --- Load templates (unchanged) ---
original_map, templates = {}, []
#  ...  (same template-loading block) ...

# ------------- MAIN LOOP -------------
while True:
    show_pattern_grid(original_map)
    selected=input("Pattern ID to unlock: ").strip()
    if selected not in original_map:
        print("Invalid ID."); continue

    # === LOG ===  prepare CSV file & counters
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/read_{selected}_{ts}.csv"
    csv_f = open(logfile, "w", newline="")
    logger = csv.writer(csv_f)
    logger.writerow(["frame","tp","fp","contour_ms","match_ms",
                     "total_ms","fps"])

    frame_idx = 0
    tp = fp = 0
    latency_samples = []

    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam"); break

    prev=time.time(); hold=0; detection_start=None

    print("Press 'r' to restart, 'q' to quit.")
    while True:
        loop_start=time.time()
        ret, frame=cap.read()
        if not ret: break
        frame_idx+=1

        # --- (original preprocessing + contour + matching) -------------
        h0,w0=frame.shape[:2]; new_h=int(h0*FRAME_WIDTH/w0)
        frm=cv2.resize(frame,(FRAME_WIDTH,new_h)); disp=frm.copy()

        t0=time.time()
        gray=cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(7,7),0)
        edges=cv2.Canny(blur,50,150)
        cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        t_contours=(time.time()-t0)*1000

        detections=[]; t_match_total=0
        for c in cnts:
            if not is_circle(c): continue
            x,y,wc,hc=cv2.boundingRect(c)
            roi=frm[y:y+hc, x:x+wc]
            _,gray_roi=preprocess(roi)
            kp2,desc2=sift.detectAndCompute(gray_roi,None)
            if desc2 is None: continue
            t1=time.time()
            pid=detect_pattern(desc2)
            t_match_total+=(time.time()-t1)*1000
            if pid: detections.append((x,y,wc,hc,pid))

        # --- decision logic (unchanged) -------------------------------
        correct_seen = any(pid==selected for *_,pid in detections)
        if correct_seen:
            tp += 1
            if detection_start is None: detection_start=time.time()
            hold=HOLD_FRAMES_MAX
        else:
            hold=max(hold-1,0)
            if hold==0: detection_start=None
            # count false negative only if the tag should be visible?
        # false positives = detections of other IDs
        fp += sum(1 for *_,pid in detections if pid!=selected)

        # unlock check
        if detection_start and time.time()-detection_start>=UNLOCK_DELAY:
            play_unlock_animation(disp,original_map[selected]); break

        # draw, overlay, FPS  (unchanged)
        total_latency=(time.time()-loop_start)*1000
        now=time.time(); fps=1.0/(now-prev); prev=now
        latency_samples.append(total_latency)

        # === LOG ===  write this frame
        logger.writerow([frame_idx, int(correct_seen),
                         sum(1 for *_,pid in detections if pid!=selected),
                         f"{t_contours:.2f}", f"{t_match_total:.2f}",
                         f"{total_latency:.2f}", f"{fps:.2f}"])

        #  (unchanged display & key handling)
        cv2.putText(disp,f"Contour:{t_contours:.1f}ms "
                         f"Match:{t_match_total:.1f}ms "
                         f"Total:{total_latency:.1f}ms",
                    (10,new_h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        cv2.putText(disp,f"FPS:{fps:.1f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
        cv2.imshow("Match",disp)
        cv2.imshow("Template",original_map[selected] if detections
                    else np.zeros_like(original_map[selected]))
        key=cv2.waitKey(1)&0xFF
        if key==ord('r'): break
        if key==ord('q'):
            cap.release(); cv2.destroyAllWindows(); csv_f.close(); exit(0)

    # === LOG ===  run summary
    cap.release(); cv2.destroyAllWindows()
    fn = frame_idx - tp             # counted as FN when tag missed
    acc = 100*tp/max(tp+fn,1)
    mean_lat = statistics.mean(latency_samples) if latency_samples else 0
    sd_lat   = statistics.stdev(latency_samples) if len(latency_samples)>1 else 0
    logger.writerow([])             # blank line
    logger.writerow(["#Summary"])
    logger.writerow(["frames", frame_idx])
    logger.writerow(["TP", tp])
    logger.writerow(["FN", fn])
    logger.writerow(["FP", fp])
    logger.writerow(["accuracy_%", f"{acc:.2f}"])
    logger.writerow(["mean_latency_ms", f"{mean_lat:.2f}"])
    logger.writerow(["std_latency_ms", f"{sd_lat:.2f}"])
    csv_f.close()
    print(f"Log saved → {logfile}")