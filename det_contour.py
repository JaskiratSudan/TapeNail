import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# ============= CONFIGURATION PARAMETERS =============
# Detection Confidence Parameters
CONFIDENCE_PARAMS = {
    'min_confidence': 0.10,     # Lowered minimum confidence for testing
    'min_match_percentage': 10, # Lowered minimum match percentage for testing
    'max_distance': 100,        # Maximum allowed distance between matched points
    'ransac_threshold': 5.0,    # Increased RANSAC threshold for more tolerance
    'ransac_confidence': 0.70   # Slightly reduced RANSAC confidence
}

# Pattern Colors (BGR format)
PATTERN_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Dark Blue
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Red
    (128, 128, 0)   # Dark Cyan
]

# SIFT Feature Detection Parameters
SIFT_PARAMS = {
    'nfeatures': 500,           # Reduced number of features for faster processing
    'nOctaveLayers': 2,        # Reduced layers for faster processing
    'contrastThreshold': 0.05,  # Slightly increased to reduce features
    'edgeThreshold': 12,       # Increased to reduce edge features
    'sigma': 1.6               # Gaussian sigma
}

# Template Processing Parameters
TEMPLATE_PARAMS = {
    'max_dimension': 200,      # Reduced maximum dimension for faster processing
    'min_matches': 8,          # Reduced minimum matches
    'match_ratio': 0.80,       # Ratio for Lowe's ratio test
    'downscale_factor': 0.5    # Downscale factor for faster processing
}

# Frame Processing Parameters
FRAME_PARAMS = {
    'width': 640,              # Frame width
    'height': 480,             # Frame height
    'fps': 30,                 # Target FPS
    'skip_frames': 3,          # Increased frame skip for better performance
    'buffer_size': 2,          # Reduced buffer size
    'roi_margin': 50,          # Margin for ROI processing
    'downscale_factor': 0.5    # Downscale factor for processing
}

# Feature Matching Parameters
MATCHING_PARAMS = {
    'trees': 4,                # Reduced number of trees
    'checks': 32,              # Reduced number of checks
    'k': 2                     # Number of nearest neighbors
}

# Threading Parameters
THREADING_PARAMS = {
    'max_workers': 4,          # Maximum number of parallel threads
    'chunk_size': 2            # Number of templates to process per thread
}

# ============= END CONFIGURATION =============

# Initialize SIFT with configured parameters
sift = cv2.SIFT_create(**SIFT_PARAMS)

class Template:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.image = cv2.imread(path)
        if self.image is not None:
            # Resize template to a reasonable size for faster processing
            h, w = self.image.shape[:2]
            if max(h, w) > TEMPLATE_PARAMS['max_dimension']:
                scale = TEMPLATE_PARAMS['max_dimension'] / max(h, w)
                self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
            
            # Create downscaled version for faster processing
            self.downscaled = cv2.resize(self.image, None, 
                                       fx=TEMPLATE_PARAMS['downscale_factor'],
                                       fy=TEMPLATE_PARAMS['downscale_factor'])
            
            self.gray = cv2.cvtColor(self.downscaled, cv2.COLOR_BGR2GRAY)
            self.kp, self.des = sift.detectAndCompute(self.gray, None)
            
            # Store original dimensions for scaling back
            self.original_h, self.original_w = self.image.shape[:2]
        else:
            print(f"Error: Unable to load template from {path}")
            self.kp = None
            self.des = None

class PatternClass:
    def __init__(self, name):
        self.name = name
        self.templates = []
        self.template_frames = []
        self.frame = None
        self.color = PATTERN_COLORS[len(self.templates) % len(PATTERN_COLORS)]

class TemplateSelector:
    def __init__(self, master):
        self.master = master
        self.master.title("Pattern Detector")
        self.master.geometry("800x600")

        self.patterns = {}  # Dictionary to store pattern classes
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create template selection frame
        self.template_frame = ttk.LabelFrame(self.main_frame, text="Pattern Templates")
        self.template_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollable frame for patterns
        self.canvas = tk.Canvas(self.template_frame)
        self.scrollbar = ttk.Scrollbar(self.template_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Add pattern button
        self.add_pattern_button = ttk.Button(self.main_frame, text="Add New Pattern", command=self.add_pattern)
        self.add_pattern_button.pack(pady=10)

        # Start button
        self.start_button = ttk.Button(self.main_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

    def add_pattern(self):
        # Create a new pattern class
        pattern_name = f"Pattern {len(self.patterns) + 1}"
        pattern = PatternClass(pattern_name)
        self.patterns[pattern_name] = pattern

        # Create frame for pattern
        pattern.frame = ttk.LabelFrame(self.scrollable_frame, text=pattern_name)
        pattern.frame.pack(fill=tk.X, padx=5, pady=5)

        # Add template button for this pattern
        add_template_btn = ttk.Button(pattern.frame, text="Add Template", 
                                    command=lambda p=pattern: self.add_template(p))
        add_template_btn.pack(pady=5)

        # Add remove pattern button
        remove_pattern_btn = ttk.Button(pattern.frame, text="Remove Pattern", 
                                      command=lambda p=pattern: self.remove_pattern(p))
        remove_pattern_btn.pack(pady=5)

        # Enable start button if we have at least one pattern
        self.start_button.config(state=tk.NORMAL)

    def add_template(self, pattern):
        template_paths = filedialog.askopenfilenames(
            initialdir="images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if template_paths:
            # Create frame for template count
            template_frame = ttk.Frame(pattern.frame)
            template_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Add template count label
            count_label = ttk.Label(template_frame, text=f"Templates: {len(pattern.templates)}")
            count_label.pack(side=tk.LEFT, padx=5)
            
            # Add remove button
            remove_btn = ttk.Button(template_frame, text="Remove All", 
                                  command=lambda f=template_frame: self.remove_all_templates(pattern, f))
            remove_btn.pack(side=tk.RIGHT, padx=5)
            
            # Add templates
            for template_path in template_paths:
                template = Template(template_path, pattern.name)
                if template.kp is not None:
                    pattern.templates.append(template)
                    count_label.config(text=f"Templates: {len(pattern.templates)}")
            
            pattern.template_frames.append(template_frame)

    def remove_all_templates(self, pattern, frame):
        pattern.templates.clear()
        frame.destroy()
        pattern.template_frames.remove(frame)

    def remove_pattern(self, pattern):
        pattern.frame.destroy()
        del self.patterns[pattern.name]
        
        # Disable start button if no patterns left
        if not self.patterns:
            self.start_button.config(state=tk.DISABLED)

    def start_detection(self):
        if self.patterns:
            self.master.destroy()
            # Flatten templates list for detection
            all_templates = []
            for pattern in self.patterns.values():
                all_templates.extend(pattern.templates)
            run_detection(all_templates)

def process_template_chunk(args):
    templates, frame, kp_frame, des_frame, flann = args
    results = []
    
    for template in templates:
        if template.kp is None or template.des is None:
            continue
            
        matches = flann.knnMatch(template.des, des_frame, k=MATCHING_PARAMS['k'])
        
        good_matches = []
        for m, n in matches:
            if m.distance < TEMPLATE_PARAMS['match_ratio'] * n.distance:
                good_matches.append(m)

        match_percentage = (len(good_matches) / len(template.des)) * 100 if len(template.des) > 0 else 0
        
        if len(good_matches) > TEMPLATE_PARAMS['min_matches']:
            src_pts = np.float32([template.kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Scale points back to original size
            src_pts *= (1.0 / TEMPLATE_PARAMS['downscale_factor'])
            dst_pts *= (1.0 / FRAME_PARAMS['downscale_factor'])
            
            # Use findHomography instead of estimateAffinePartial2D for better results
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                       CONFIDENCE_PARAMS['ransac_threshold'],
                                       maxIters=2000,
                                       confidence=CONFIDENCE_PARAMS['ransac_confidence'])
            
            if M is not None and match_percentage >= CONFIDENCE_PARAMS['min_match_percentage']:
                # Define template corners
                h, w = template.original_h, template.original_w
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                
                # Transform points using homography
                dst = cv2.perspectiveTransform(pts, M)
                
                # Calculate confidence score
                confidence = (match_percentage / 100.0) * (np.sum(mask) / len(mask))
                
                if confidence >= CONFIDENCE_PARAMS['min_confidence']:
                    # Convert corners to rectangle format (x, y, width, height)
                    corners = dst.reshape(-1, 2)
                    x_coords = corners[:, 0]
                    y_coords = corners[:, 1]
                    
                    # Calculate rectangle parameters
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - x)
                    height = int(max(y_coords) - y)
                    
                    results.append((template, match_percentage, (x, y, width, height), confidence))
    
    return results

def run_detection(templates):
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_PARAMS['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_PARAMS['height'])
    cap.set(cv2.CAP_PROP_FPS, FRAME_PARAMS['fps'])
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=MATCHING_PARAMS['trees'])
    search_params = dict(checks=MATCHING_PARAMS['checks'])
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Initialize thread pool for parallel processing
    executor = ThreadPoolExecutor(max_workers=THREADING_PARAMS['max_workers'])
    
    frame_count = 0
    last_detections = {}  # Store last successful detections
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Skip frames for better performance
        if frame_count % FRAME_PARAMS['skip_frames'] != 0:
            continue
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (FRAME_PARAMS['width'], FRAME_PARAMS['height']))
        
        # Process frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        
        # Downscale frame for faster processing
        downscaled = cv2.resize(gray_frame, None, 
                              fx=FRAME_PARAMS['downscale_factor'],
                              fy=FRAME_PARAMS['downscale_factor'])
        
        kp_frame, des_frame = sift.detectAndCompute(downscaled, None)
        
        if des_frame is not None:
            # Process templates in chunks for better parallelization
            chunk_size = THREADING_PARAMS['chunk_size']
            template_chunks = [templates[i:i + chunk_size] for i in range(0, len(templates), chunk_size)]
            
            args = [(chunk, frame, kp_frame, des_frame, flann) for chunk in template_chunks]
            chunk_results = list(executor.map(process_template_chunk, args))
            
            # Flatten results
            results = [item for chunk in chunk_results for item in chunk]
            
            # Group results by pattern name
            pattern_results = {}
            for template, match_percentage, dst, confidence in results:
                if dst is not None:
                    if template.name not in pattern_results:
                        pattern_results[template.name] = []
                    pattern_results[template.name].append((match_percentage, dst, confidence))
            
            # Draw results (use best match for each pattern)
            y_offset = 30
            for pattern_name, matches in pattern_results.items():
                # Sort by confidence and use the best match
                best_match = max(matches, key=lambda x: x[2])
                match_percentage, rect, confidence = best_match
                
                # Store last successful detection
                last_detections[pattern_name] = (rect, confidence)
                
                # Get pattern color and adjust brightness based on confidence
                pattern_color = PATTERN_COLORS[list(pattern_results.keys()).index(pattern_name) % len(PATTERN_COLORS)]
                color = tuple(int(c * confidence) for c in pattern_color)
                
                # Draw rectangle
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw text with pattern color
                cv2.putText(frame, f"{pattern_name}: {match_percentage:.1f}% ({confidence:.2f})", 
                          (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, pattern_color, 2)
                y_offset += 30
            
            # Draw last known detections for patterns not detected in current frame
            for pattern_name, (rect, confidence) in last_detections.items():
                if pattern_name not in pattern_results:
                    pattern_color = PATTERN_COLORS[list(last_detections.keys()).index(pattern_name) % len(PATTERN_COLORS)]
                    color = tuple(int(c * confidence * 0.5) for c in pattern_color)  # Dimmed color for old detections
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add debug information
            cv2.putText(frame, f"Features: {len(kp_frame)}", (10, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Frame', frame)
        else:
            cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    executor.shutdown()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = TemplateSelector(root)
    root.mainloop()
