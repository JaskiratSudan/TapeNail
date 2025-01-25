import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

sift = cv2.SIFT_create(nfeatures=0)

def load_template(template_path):
    template = cv2.imread(template_path)
    if template is None:
        print(f"Error: Unable to load template from {template_path}")
        return None, None, None
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    kp_template, des_template = sift.detectAndCompute(gray_template, None)
    return template, kp_template, des_template

class TemplateSelector:
    def __init__(self, master):
        self.master = master
        self.master.title("Template Selector")
        self.master.geometry("300x150")

        self.template_path = None
        self.template_image = None

        self.select_button = tk.Button(self.master, text="Select Template", command=self.select_template)
        self.select_button.pack(pady=20)

        self.start_button = tk.Button(self.master, text="Start Detection", command=self.start_detection, state=tk.DISABLED)
        self.start_button.pack(pady=10)

    def select_template(self):
        self.template_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.template_path:
            self.template_image = Image.open(self.template_path)
            self.template_image.thumbnail((100, 100))
            self.template_photo = ImageTk.PhotoImage(self.template_image)
            
            if hasattr(self, 'template_label'):
                self.template_label.config(image=self.template_photo)
            else:
                self.template_label = tk.Label(self.master, image=self.template_photo)
                self.template_label.pack()

            self.start_button.config(state=tk.NORMAL)

    def start_detection(self):
        self.master.destroy()
        run_detection(self.template_path)

def run_detection(template_path):
    template, kp_template, des_template = load_template(template_path)

    if template is not None:
        cap = cv2.VideoCapture(0)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        frame_count = 0
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            frame_count += 1
            if not ret:
                print("Failed to grab frame")
                break
            
            if frame_count % 2 != 0:
                continue
            
            frame = cv2.resize(frame, (800, 600))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.equalizeHist(gray_frame)
            
            kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
            
            if des_frame is not None:
                matches = flann.knnMatch(des_template, des_frame, k=2)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.80 * n.distance:
                        good_matches.append(m)

                # Calculate matching percentage
                match_percentage = (len(good_matches) / len(des_template)) * 100 if len(des_template) > 0 else 0
                
                if len(good_matches) > 10:
                    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    
                    if M is not None:
                        h, w = template.shape[:2]
                        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.transform(pts, M)
                        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show similarity score 
                cv2.putText(frame, f"Similarity: {match_percentage:.2f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                result = cv2.drawMatches(template, kp_template, frame, kp_frame, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow('Matched Frame', result)
            else:
                cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Template loading failed. Please check the file path and integrity.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TemplateSelector(root)
    root.mainloop()
