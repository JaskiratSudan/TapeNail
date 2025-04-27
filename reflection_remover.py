import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class ReflectionRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("Reflection Remover")
        self.root.geometry("1200x800")
        
        # Variables
        self.original_image = None
        self.processed_image = None
        self.photo = None
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for controls
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right panel for image display
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self.right_panel, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create controls
        self.create_controls()
        
    def create_controls(self):
        # Load image button
        ttk.Button(self.left_panel, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Method selection
        method_frame = ttk.LabelFrame(self.left_panel, text="Reflection Removal Methods")
        method_frame.pack(fill=tk.X, pady=5)
        
        # Create method buttons
        methods = [
            ("CLAHE", self.apply_clahe),
            ("Bilateral Filter", self.apply_bilateral),
            ("Gaussian Blur", self.apply_gaussian),
            ("Median Blur", self.apply_median),
            ("Non-Local Means", self.apply_nlm),
            ("Homomorphic Filtering", self.apply_homomorphic),
            ("Adaptive Threshold", self.apply_adaptive_threshold)
        ]
        
        for method_name, method_func in methods:
            ttk.Button(method_frame, text=method_name, command=method_func).pack(fill=tk.X, pady=2)
        
        # Parameter controls
        self.param_frame = ttk.LabelFrame(self.left_panel, text="Parameters")
        self.param_frame.pack(fill=tk.X, pady=5)
        
        # CLAHE parameters
        self.clahe_clip_limit = tk.DoubleVar(value=2.0)
        self.clahe_grid_size = tk.IntVar(value=8)
        
        # Bilateral parameters
        self.bilateral_d = tk.IntVar(value=9)
        self.bilateral_sigma_color = tk.IntVar(value=75)
        self.bilateral_sigma_space = tk.IntVar(value=75)
        
        # Gaussian parameters
        self.gaussian_kernel = tk.IntVar(value=5)
        self.gaussian_sigma = tk.DoubleVar(value=0)
        
        # Median parameters
        self.median_kernel = tk.IntVar(value=5)
        
        # NLM parameters
        self.nlm_h = tk.DoubleVar(value=10)
        self.nlm_template = tk.IntVar(value=7)
        self.nlm_search = tk.IntVar(value=21)
        
        # Homomorphic parameters
        self.homomorphic_gamma_low = tk.DoubleVar(value=0.5)
        self.homomorphic_gamma_high = tk.DoubleVar(value=2.0)
        self.homomorphic_cutoff = tk.DoubleVar(value=30)
        
        # Adaptive threshold parameters
        self.adaptive_block = tk.IntVar(value=11)
        self.adaptive_c = tk.IntVar(value=2)
        
        # Create parameter controls
        self.create_parameter_controls()
        
        # Reset button
        ttk.Button(self.left_panel, text="Reset Image", command=self.reset_image).pack(fill=tk.X, pady=5)
        
        # Save button
        ttk.Button(self.left_panel, text="Save Processed Image", command=self.save_image).pack(fill=tk.X, pady=5)
    
    def create_parameter_controls(self):
        # CLAHE parameters
        ttk.Label(self.param_frame, text="CLAHE:").pack(anchor=tk.W)
        ttk.Label(self.param_frame, text="Clip Limit").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=1, to=10, variable=self.clahe_clip_limit,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Grid Size").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=2, to=16, variable=self.clahe_grid_size,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Bilateral parameters
        ttk.Label(self.param_frame, text="Bilateral:").pack(anchor=tk.W)
        ttk.Label(self.param_frame, text="Diameter").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=1, to=15, variable=self.bilateral_d,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Sigma Color").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=10, to=200, variable=self.bilateral_sigma_color,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Sigma Space").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=10, to=200, variable=self.bilateral_sigma_space,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Gaussian parameters
        ttk.Label(self.param_frame, text="Gaussian:").pack(anchor=tk.W)
        ttk.Label(self.param_frame, text="Kernel Size").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=1, to=15, variable=self.gaussian_kernel,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Sigma").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=0, to=10, variable=self.gaussian_sigma,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Median parameters
        ttk.Label(self.param_frame, text="Median:").pack(anchor=tk.W)
        ttk.Label(self.param_frame, text="Kernel Size").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=1, to=15, variable=self.median_kernel,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # NLM parameters
        ttk.Label(self.param_frame, text="Non-Local Means:").pack(anchor=tk.W)
        ttk.Label(self.param_frame, text="Filter Strength").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=1, to=30, variable=self.nlm_h,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Template Size").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=3, to=15, variable=self.nlm_template,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Search Size").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=5, to=30, variable=self.nlm_search,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Homomorphic parameters
        ttk.Label(self.param_frame, text="Homomorphic:").pack(anchor=tk.W)
        ttk.Label(self.param_frame, text="Gamma Low").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=0.1, to=1.0, variable=self.homomorphic_gamma_low,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Gamma High").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=1.0, to=5.0, variable=self.homomorphic_gamma_high,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="Cutoff").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=1, to=100, variable=self.homomorphic_cutoff,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Adaptive threshold parameters
        ttk.Label(self.param_frame, text="Adaptive Threshold:").pack(anchor=tk.W)
        ttk.Label(self.param_frame, text="Block Size").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=3, to=31, variable=self.adaptive_block,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.param_frame, text="C").pack(anchor=tk.W)
        ttk.Scale(self.param_frame, from_=0, to=20, variable=self.adaptive_c,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            self.update_display()
    
    def update_display(self, image=None):
        if image is None:
            image = self.processed_image
            
        if image is None:
            return
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate aspect ratio
            img_width, img_height = pil_image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            
            # Resize image
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width//2, canvas_height//2,
                image=self.photo, anchor=tk.CENTER
            )
    
    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_display()
    
    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
    
    def apply_clahe(self):
        if self.processed_image is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit.get(),
            tileGridSize=(self.clahe_grid_size.get(), self.clahe_grid_size.get())
        )
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR
        self.processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        self.update_display()
    
    def apply_bilateral(self):
        if self.processed_image is None:
            return
            
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            self.processed_image,
            self.bilateral_d.get(),
            self.bilateral_sigma_color.get(),
            self.bilateral_sigma_space.get()
        )
        self.processed_image = filtered
        self.update_display()
    
    def apply_gaussian(self):
        if self.processed_image is None:
            return
            
        # Ensure kernel size is odd
        ksize = self.gaussian_kernel.get()
        if ksize % 2 == 0:
            ksize += 1
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            self.processed_image,
            (ksize, ksize),
            self.gaussian_sigma.get()
        )
        self.processed_image = blurred
        self.update_display()
    
    def apply_median(self):
        if self.processed_image is None:
            return
            
        # Ensure kernel size is odd
        ksize = self.median_kernel.get()
        if ksize % 2 == 0:
            ksize += 1
            
        # Apply median blur
        blurred = cv2.medianBlur(self.processed_image, ksize)
        self.processed_image = blurred
        self.update_display()
    
    def apply_nlm(self):
        if self.processed_image is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=self.nlm_h.get(),
            templateWindowSize=self.nlm_template.get(),
            searchWindowSize=self.nlm_search.get()
        )
        
        # Convert back to BGR
        self.processed_image = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        self.update_display()
    
    def apply_homomorphic(self):
        if self.processed_image is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        # Convert to float32
        gray_float = np.float32(gray)
        
        # Apply log
        log_img = np.log1p(gray_float)
        
        # Apply FFT
        fft = np.fft.fft2(log_img)
        fft_shift = np.fft.fftshift(fft)
        
        # Create filter
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        cutoff = self.homomorphic_cutoff.get()
        gamma_low = self.homomorphic_gamma_low.get()
        gamma_high = self.homomorphic_gamma_high.get()
        
        # Create filter
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if d <= cutoff:
                    mask[i, j] = gamma_low
                else:
                    mask[i, j] = gamma_high
        
        # Apply filter
        fft_shift_filtered = fft_shift * mask
        
        # Inverse FFT
        fft_ishift = np.fft.ifftshift(fft_shift_filtered)
        img_back = np.fft.ifft2(fft_ishift)
        img_back = np.abs(img_back)
        
        # Apply exp
        img_back = np.expm1(img_back)
        
        # Normalize
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)
        
        # Convert back to BGR
        self.processed_image = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        self.update_display()
    
    def apply_adaptive_threshold(self):
        if self.processed_image is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        # Ensure block size is odd
        block_size = self.adaptive_block.get()
        if block_size % 2 == 0:
            block_size += 1
            
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.adaptive_c.get()
        )
        
        # Convert back to BGR
        self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ReflectionRemover(root)
    root.mainloop() 