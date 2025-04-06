import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Configuration Parameters
CONFIDENCE_PARAMS = {
    'min_confidence': 0.1,     # Minimum confidence score for detection
    'min_match_percentage': 3, # Minimum percentage of matched features
    'max_distance': 100,        # Maximum allowed distance between matched points
    'ransac_threshold': 5.0,    # RANSAC threshold for homography estimation
    'ransac_confidence': 0.70   # RANSAC confidence level
}

SIFT_PARAMS = {
    'nfeatures': 0,            # 0 means no limit on features
    'nOctaveLayers': 3,        # Number of layers in each octave
    'contrastThreshold': 0.04,  # Lower value = more features
    'edgeThreshold': 10,       # Lower value = more edge features
    'sigma': 1.6               # Gaussian sigma
}

TEMPLATE_PARAMS = {
    'max_dimension': 200,      # Maximum dimension for template resizing
    'min_matches': 6,          # Minimum number of good matches required
    'match_ratio': 0.75,       # Ratio for Lowe's ratio test
    'downscale_factor': 0.5    # Downscale factor for faster processing
}

FRAME_PARAMS = {
    'width': 640,              # Frame width
    'height': 480,             # Frame height
    'fps': 30,                 # Target FPS
    'skip_frames': 3,          # Process every nth frame
    'downscale_factor': 0.5    # Downscale factor for processing
}

MATCHING_PARAMS = {
    'trees': 4,                # Number of trees in FLANN matcher
    'checks': 32,              # Number of checks in FLANN matcher
    'k': 2                     # Number of nearest neighbors
}

THREADING_PARAMS = {
    'max_workers': 4,          # Maximum number of parallel threads
    'chunk_size': 2            # Number of templates to process per thread
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

class Template:
    def __init__(self, path, pattern_num):
        self.path = path
        self.pattern_num = pattern_num
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

def load_templates(image_dir):
    """Load all templates from the images directory."""
    templates = []
    pattern_templates = {}  # Dictionary to store templates by pattern number
    
    # Initialize SIFT
    global sift
    sift = cv2.SIFT_create(**SIFT_PARAMS)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Group files by pattern number
    for filename in image_files:
        if filename.startswith('pat_'):
            try:
                pattern_num = int(filename.split('_')[1])
                if pattern_num not in pattern_templates:
                    pattern_templates[pattern_num] = []
                pattern_templates[pattern_num].append(filename)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse pattern number from {filename}")
    
    # Load templates for each pattern
    for pattern_num, files in pattern_templates.items():
        print(f"Loading {len(files)} templates for pattern {pattern_num}")
        for filename in files:
            template_path = os.path.join(image_dir, filename)
            template = Template(template_path, pattern_num)
            if template.kp is not None:
                templates.append(template)
    
    return templates

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
            
            # Use findHomography for better results
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
                    
                    # Add debug print
                    print(f"Pattern {template.pattern_num} detected: {match_percentage:.1f}% matches, confidence: {confidence:.2f}")
                    
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
            
            # Group results by pattern number
            pattern_results = {}
            for template, match_percentage, rect, confidence in results:
                if rect is not None:
                    if template.pattern_num not in pattern_results:
                        pattern_results[template.pattern_num] = []
                    pattern_results[template.pattern_num].append((match_percentage, rect, confidence))
            
            # Draw results (use best match for each pattern)
            y_offset = 30
            for pattern_num, matches in pattern_results.items():
                # Sort by confidence and use the best match
                best_match = max(matches, key=lambda x: x[2])
                match_percentage, rect, confidence = best_match
                
                # Store last successful detection
                last_detections[pattern_num] = (rect, confidence)
                
                # Get pattern color
                color = PATTERN_COLORS[pattern_num % len(PATTERN_COLORS)]
                
                # Draw rectangle
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw text
                cv2.putText(frame, f"Pattern {pattern_num}: {match_percentage:.1f}% ({confidence:.2f})", 
                          (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                y_offset += 30
            
            # Draw last known detections for patterns not detected in current frame
            for pattern_num, (rect, confidence) in last_detections.items():
                if pattern_num not in pattern_results:
                    color = PATTERN_COLORS[pattern_num % len(PATTERN_COLORS)]
                    color = tuple(int(c * confidence * 0.5) for c in color)  # Dimmed color for old detections
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
    # Load templates from the images directory
    templates = load_templates("cropped_multiclass_dataset")
    print(f"Loaded {len(templates)} templates")
    
    # Run detection
    run_detection(templates) 