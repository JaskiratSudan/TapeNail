import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk

# --------------------------------
# Utility Functions
# --------------------------------
def rootSIFT(des, eps=1e-7):
    """
    Apply RootSIFT to SIFT descriptors:
      1. L1-normalize each descriptor.
      2. Take the square root of each element.
    """
    des_normalized = des.copy()
    des_normalized /= (np.sum(des_normalized, axis=1, keepdims=True) + eps)
    return np.sqrt(des_normalized)

def remove_reflections(image, threshold=240, inpaint_radius=3):
    """
    Remove specular highlights/reflections using inpainting.
    Pixels above the brightness threshold are inpainted.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """
    Adjust brightness and contrast.
    Brightness is added to pixel values.
    Contrast is a multiplicative factor.
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def augment_template(image):
    """
    Generate a list of augmented images (variants) from the original template.
    Variants include the original plus slight brightness/contrast adjustments.
    """
    variants = []
    # Use the original image
    variants.append(image)
    # Generate darker and brighter variants
    variants.append(adjust_brightness_contrast(image, brightness=-30, contrast=1.0))
    variants.append(adjust_brightness_contrast(image, brightness=30, contrast=1.0))
    # Generate variant with increased contrast
    variants.append(adjust_brightness_contrast(image, brightness=0, contrast=1.2))
    variants.append(adjust_brightness_contrast(image, brightness=0, contrast=0.8))
    # You may add further augmentations (rotations, slight blurring, etc.) if needed
    return variants

# --------------------------------
# Configuration Parameters
# --------------------------------
CONFIDENCE_PARAMS = {
    'min_confidence': 0.1,
    'min_match_percentage': 15,
    'max_distance': 200,
    'ransac_threshold': 5.0,
    'ransac_confidence': 0.70
}

SIFT_PARAMS = {
    'nfeatures': 0,
    'nOctaveLayers': 4,
    'contrastThreshold': 0.03,
    'edgeThreshold': 15,
    'sigma': 1.6
}

TEMPLATE_PARAMS = {
    'max_dimension': 150,
    'min_matches': 8,
    'match_ratio': 0.75,
    'downscale_factor': 0.5
}

FRAME_PARAMS = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'skip_frames': 4,
    'downscale_factor': 0.5
}

MATCHING_PARAMS = {
    'trees': 4,
    'checks': 32,
    'k': 2,
    'ratio_threshold': 0.75,
    'min_matches': 8,
    'max_distance': 200
}

THREADING_PARAMS = {
    'max_workers': 8,
    'chunk_size': 4
}

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

# --------------------------------
# Template Class with Augmentation & RootSIFT
# --------------------------------
class Template:
    def __init__(self, path, pattern_num):
        self.path = path
        self.pattern_num = pattern_num
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            print(f"Error: Unable to load template from {path}")
            self.variants = []
            return

        # Remove reflections and resize if needed
        clean_img = remove_reflections(self.original_image)
        # Optionally resize image (here we keep original size, you can add resizing if needed)
        
        # Preprocess image with LAB conversion, CLAHE, and bilateral filtering
        lab = cv2.cvtColor(clean_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        lab = cv2.bilateralFilter(lab, 5, 50, 50)
        preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Generate augmented variants
        aug_variants = augment_template(preprocessed)
        
        # For each variant, compute SIFT keypoints and RootSIFT descriptors.
        # Variants with too few keypoints are discarded.
        self.variants = []
        for img in aug_variants:
            kp, des = sift.detectAndCompute(img, None)
            if des is not None and len(kp) > 0:
                des = rootSIFT(des)
                self.variants.append({'image': img, 'kp': kp, 'des': des})
        
        self.original_h, self.original_w = preprocessed.shape[:2]

# --------------------------------
# Visualization Helper: Show Template Preview
# --------------------------------
def show_template_preview(templates):
    """
    Display one representative variant per pattern along with the SIFT keypoints.
    """
    pattern_templates = {}
    for template in templates:
        if template.pattern_num not in pattern_templates and len(template.variants) > 0:
            pattern_templates[template.pattern_num] = template.variants[0]
    
    n_patterns = len(pattern_templates)
    grid_size = int(np.ceil(np.sqrt(n_patterns)))
    
    # Get screen size and adjust figure size
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    # Calculate figure size to fit screen (80% of screen size)
    fig_width = screen_width * 0.8 / 100  # Convert to inches (matplotlib uses inches)
    fig_height = screen_height * 0.8 / 100
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(grid_size, grid_size, figure=fig)
    
    for i, (pattern_num, variant) in enumerate(pattern_templates.items()):
        row = i // grid_size
        col = i % grid_size
        ax = fig.add_subplot(gs[row, col])
        img_rgb = cv2.cvtColor(variant['image'], cv2.COLOR_BGR2RGB)
        img_with_kp = cv2.drawKeypoints(
            img_rgb, variant['kp'], None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        ax.imshow(img_with_kp)
        ax.set_title(f"Pattern {pattern_num}\nFeatures: {len(variant['kp'])}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# --------------------------------
# Unlock Animation and Display Functions
# --------------------------------
def create_unlock_animation(frame, pattern_num, rect):
    x, y, w, h = rect
    center_x, center_y = x + w // 2, y + h // 2
    anim_frame = frame.copy()
    radius = max(w, h) // 2
    color = PATTERN_COLORS[pattern_num % len(PATTERN_COLORS)]
    thickness = 3
    # Draw a simple check mark as animation
    check_size = radius // 2
    pt1 = (center_x - check_size, center_y)
    pt2 = (center_x - check_size // 2, center_y + check_size)
    pt3 = (center_x + check_size, center_y - check_size)
    cv2.line(anim_frame, pt1, pt2, color, thickness * 2)
    cv2.line(anim_frame, pt2, pt3, color, thickness * 2)
    cv2.imshow("Frame", anim_frame)
    cv2.waitKey(500)

def show_unlocked_text(frame):
    text_frame = frame.copy()
    height, width = text_frame.shape[:2]
    overlay = text_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, text_frame, 0.5, 0, text_frame)
    font_scale = 3.0
    thickness = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize("Unlocked", font, font_scale, thickness)
    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2
    cv2.putText(text_frame, "Unlocked", (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(text_frame, "Unlocked", (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    cv2.imshow("Frame", text_frame)
    print("System Unlocked! Press any key to continue...")
    cv2.waitKey(0)

# --------------------------------
# Template Loading Function
# --------------------------------
def load_templates(image_dir):
    templates = []
    pattern_templates = {}
    
    global sift
    sift = cv2.SIFT_create(**SIFT_PARAMS)
    
    # Get all shape folders
    shape_folders = ["circle", "square", "triangle"]
    
    for shape in shape_folders:
        shape_dir = os.path.join(image_dir, shape)
        if not os.path.exists(shape_dir):
            continue
            
        # Get all PNG files in the shape folder
        image_files = [f for f in os.listdir(shape_dir) if f.lower().endswith('.png')]
        
        for filename in image_files:
            try:
                # Extract pattern number from filename (e.g., "pat_01_c.png" -> 1)
                pattern_num = int(filename.split('_')[1])
                template_path = os.path.join(shape_dir, filename)
                tpl = Template(template_path, pattern_num)
                # Only add if at least one augmented variant is valid
                if tpl.variants:
                    templates.append(tpl)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse pattern number from {filename}")
                continue
    
    show_template_preview(templates)
    return templates

# --------------------------------
# Template Chunk Processing (Parallelized)
# --------------------------------
def process_template_chunk(args):
    """
    Process a chunk of templates (each having several augmented variants) against the current frame.
    Returns matching results (if found) for each pattern.
    """
    templates, frame, flann = args
    results = []
    # Preprocess the frame similar to how we preprocess templates:
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab_frame = cv2.merge([l, a, b])
    lab_frame = cv2.bilateralFilter(lab_frame, 5, 50, 50)
    proc_frame = cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)
    
    kp_frame, des_frame = sift.detectAndCompute(proc_frame, None)
    if des_frame is None or len(kp_frame) == 0:
        return results
    des_frame = rootSIFT(des_frame)
    
    # For each template, check all variants:
    for tpl in templates:
        for variant in tpl.variants:
            matches = flann.knnMatch(variant["des"], des_frame, k=MATCHING_PARAMS["k"])
            good_matches = [m for m, n in matches if m.distance < MATCHING_PARAMS["ratio_threshold"] * n.distance]
            if len(good_matches) >= MATCHING_PARAMS["min_matches"]:
                src_pts = np.float32([ variant["kp"][m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                
                # Scale points if necessary (here we assume consistent processing)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                             ransacReprojThreshold=CONFIDENCE_PARAMS["ransac_threshold"],
                                             confidence=CONFIDENCE_PARAMS["ransac_confidence"],
                                             maxIters=1000)
                if H is not None and mask is not None and np.sum(mask) >= MATCHING_PARAMS["min_matches"]:
                    match_percentage = (np.sum(mask) / len(variant["kp"])) * 100
                    if match_percentage >= CONFIDENCE_PARAMS["min_match_percentage"]:
                        h, w = tpl.original_h, tpl.original_w
                        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        transformed_corners = cv2.perspectiveTransform(corners, H)
                        confidence = match_percentage / 100.0
                        x, y = np.min(transformed_corners, axis=0)[0]
                        w_rect = np.max(transformed_corners[:, :, 0]) - x
                        h_rect = np.max(transformed_corners[:, :, 1]) - y
                        results.append({
                            "pattern_num": tpl.pattern_num,
                            "rect": (int(x), int(y), int(w_rect), int(h_rect)),
                            "confidence": confidence,
                            "match_percentage": match_percentage,
                        })
                        # Break after one positive match per template variant
                        break
    return results

# --------------------------------
# Main Detection Function
# --------------------------------
def run_detection(templates):
    # List available cameras
    print("\nAvailable cameras:")
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Camera {i}: Available")
                available_cameras.append(i)
            else:
                print(f"Camera {i}: Connected but failed to read frame")
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    if not available_cameras:
        print("\nNo cameras found! Please check your camera connection and drivers.")
        return
    
    camera_index = None
    while camera_index is None:
        try:
            camera_index = int(input("\nEnter camera index to use (0 for default): "))
            if camera_index not in available_cameras:
                print(f"Camera {camera_index} is not available. Choose from: {available_cameras}")
                camera_index = None
        except ValueError:
            print("Please enter a valid number")
    
    print(f"Using camera {camera_index}")
    max_retries = 3
    retry_count = 0
    cap = None
    while retry_count < max_retries:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                break
            else:
                print(f"Camera {camera_index} opened but failed to read frame. Retrying...")
                cap.release()
        else:
            print(f"Failed to open camera {camera_index}. Retrying...")
        retry_count += 1
        time.sleep(1)
    
    if not cap or not cap.isOpened():
        print("\nFailed to initialize camera after multiple attempts.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_PARAMS["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_PARAMS["height"])
    cap.set(cv2.CAP_PROP_FPS, FRAME_PARAMS["fps"])
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nCamera settings:\nResolution: {actual_width}x{actual_height}\nFPS: {actual_fps}")
    
    ret, frame = cap.read()
    if not ret:
        print("\nFailed to read initial frame from camera.")
        cap.release()
        return
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=16)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    executor = ThreadPoolExecutor(max_workers=THREADING_PARAMS["max_workers"])
    
    frame_count = 0
    last_detections = {}
    unlock_pattern = None
    unlock_detection_start_time = None
    unlock_detection_duration = 5.0
    unlock_animation_played = False
    
    print("\nSelect a pattern number to use as the unlock pattern:")
    pattern_nums = sorted({tpl.pattern_num for tpl in templates})
    for num in pattern_nums:
        print(f"Pattern {num}")
    while unlock_pattern is None:
        try:
            unlock_pattern = int(input("Enter pattern number: "))
            if unlock_pattern not in pattern_nums:
                print(f"Invalid pattern number. Please choose from: {pattern_nums}")
                unlock_pattern = None
        except ValueError:
            print("Please enter a valid number")
    print(f"Pattern {unlock_pattern} selected as unlock pattern")
    
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            consecutive_failures += 1
            print(f"Failed to grab frame (attempt {consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                print("\nToo many consecutive frame grab failures. Check your camera connection.")
                break
            time.sleep(0.1)
            continue
        consecutive_failures = 0
        if frame_count % FRAME_PARAMS["skip_frames"] != 0:
            continue
        
        frame = remove_reflections(frame)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.addWeighted(frame, 1.2, frame, -0.2, 0)
        frame = cv2.resize(frame, (FRAME_PARAMS["width"], FRAME_PARAMS["height"]))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        downscaled = cv2.resize(gray_frame, None, fx=FRAME_PARAMS["downscale_factor"], fy=FRAME_PARAMS["downscale_factor"])
        
        kp_frame, des_frame = sift.detectAndCompute(downscaled, None)
        
        if des_frame is not None:
            args = [(templates, frame, flann)]
            chunk_results = list(executor.map(process_template_chunk, args))
            results = [item for sublist in chunk_results for item in sublist]
            
            pattern_results = {}
            for result in results:
                pattern_results.setdefault(result["pattern_num"], []).append((result["match_percentage"], result["rect"], result["confidence"]))
            
            unlock_detected = False
            unlock_rect = None
            y_offset = 30
            for pattern_num, matches in pattern_results.items():
                best_match = max(matches, key=lambda x: x[2])
                match_percentage, rect, confidence = best_match
                last_detections[pattern_num] = (rect, confidence)
                if pattern_num == unlock_pattern:
                    unlock_detected = True
                    unlock_rect = rect
                color = PATTERN_COLORS[pattern_num % len(PATTERN_COLORS)]
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, f"Pattern {pattern_num}: {match_percentage:.1f}% ({confidence:.2f})", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                y_offset += 30
            
            for pattern_num, (rect, confidence) in last_detections.items():
                if pattern_num not in pattern_results:
                    color = PATTERN_COLORS[pattern_num % len(PATTERN_COLORS)]
                    dimmed = tuple(int(c * confidence * 0.5) for c in color)
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), dimmed, 2)
            
            current_time = time.time()
            if unlock_detected:
                if unlock_detection_start_time is None:
                    unlock_detection_start_time = current_time
                    print(f"Unlock pattern detected. Waiting {unlock_detection_duration} seconds...")
                if not unlock_animation_played and (current_time - unlock_detection_start_time) >= unlock_detection_duration:
                    print("Unlock pattern visible for 5 seconds!")
                    unlock_animation_played = True
                    create_unlock_animation(frame, unlock_pattern, unlock_rect)
                    show_unlocked_text(frame)
            else:
                unlock_detection_start_time = None
            if unlock_detection_start_time is not None and not unlock_animation_played:
                remaining_time = unlock_detection_duration - (current_time - unlock_detection_start_time)
                if remaining_time > 0:
                    cv2.putText(frame, f"Unlocking in: {remaining_time:.1f}s", (10, y_offset + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Features: {len(kp_frame)}", (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
        else:
            cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    executor.shutdown()
    cap.release()
    cv2.destroyAllWindows()

# --------------------------------
# Main Execution
# --------------------------------
if __name__ == "__main__":
    templates = load_templates("patterns")  # Changed from "captured_shapes" to "patterns"
    print(f"Loaded {len(templates)} templates")
    run_detection(templates)