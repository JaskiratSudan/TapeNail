import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import glob

class YOLOLabelEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Label Editor")
        self.root.geometry("1400x800")
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.current_image_path = None
        self.current_label_path = None
        self.current_image_index = 0
        self.image_files = []
        self.label_files = []
        self.current_label = None
        self.class_examples = {}  # Dictionary to store example images for each class
        
        # Create main container
        self._create_widgets()
        
    def _create_widgets(self):
        # Main content area with three panels
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left panel for current image
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True)
        
        # Dataset controls
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill="x", pady=5)
        ttk.Label(controls_frame, text="Dataset:").pack(side="left", padx=5)
        ttk.Entry(controls_frame, textvariable=self.dataset_path, width=50).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Browse", command=self._browse_dataset).pack(side="left", padx=5)
        
        # Navigation
        nav_frame = ttk.Frame(left_panel)
        nav_frame.pack(fill="x", pady=5)
        ttk.Button(nav_frame, text="Previous (←)", command=self._prev_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Next (→)", command=self._next_image).pack(side="left", padx=5)
        self.image_info = ttk.Label(nav_frame, text="")
        self.image_info.pack(side="right", padx=5)
        
        # Current image display
        self.image_label = ttk.Label(left_panel)
        self.image_label.pack(fill="both", expand=True)
        
        # Middle panel for class example
        middle_panel = ttk.LabelFrame(content_frame, text="Example of Current Class", padding=10)
        middle_panel.pack(side="left", fill="both", padx=10)
        
        self.example_label = ttk.Label(middle_panel)
        self.example_label.pack(fill="both", expand=True)
        
        # Right panel for controls
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side="right", fill="y", padx=10)
        
        # Class ID input
        label_frame = ttk.LabelFrame(right_panel, text="Label Editor", padding=10)
        label_frame.pack(fill="x", pady=5)
        
        ttk.Label(label_frame, text="Class ID:").pack(anchor="w")
        self.class_id = ttk.Entry(label_frame)
        self.class_id.pack(fill="x", pady=5)
        
        ttk.Button(label_frame, text="Save Changes (Enter)", command=self._save_changes).pack(fill="x", pady=5)
        
        # Bind keyboard shortcuts
        self.root.bind('<Left>', lambda e: self._prev_image())
        self.root.bind('<Right>', lambda e: self._next_image())
        self.root.bind('<Return>', lambda e: self._save_changes())
    
    def _browse_dataset(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path.set(folder)
            self._load_dataset()
    
    def _load_dataset(self):
        dataset_path = self.dataset_path.get()
        if not dataset_path:
            return
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(dataset_path, "images", "*.jpg")))
        self.label_files = sorted(glob.glob(os.path.join(dataset_path, "labels", "*.txt")))
        
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the dataset!")
            return
        
        print(f"Found {len(self.image_files)} images and {len(self.label_files)} labels")
        
        # Load example images for each class
        self._load_class_examples()
        
        # Display first image
        self.current_image_index = 0
        self._display_current_image()
    
    def _load_class_examples(self):
        """Load one example image for each class from the dataset"""
        self.class_examples.clear()
        
        # Find first image for each class
        for label_file in self.label_files:
            try:
                with open(label_file, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        if class_id not in self.class_examples:
                            # Get corresponding image file
                            image_file = os.path.join(
                                os.path.dirname(os.path.dirname(label_file)),
                                "images",
                                os.path.splitext(os.path.basename(label_file))[0] + ".jpg"
                            )
                            if os.path.exists(image_file):
                                # Load and resize image
                                image = Image.open(image_file)
                                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                                photo = ImageTk.PhotoImage(image)
                                self.class_examples[class_id] = photo
            except Exception as e:
                print(f"Error loading example for class {class_id}: {str(e)}")
    
    def _display_current_image(self):
        if not self.image_files:
            return
        
        # Update image info
        self.image_info.config(text=f"Image {self.current_image_index + 1}/{len(self.image_files)}")
        
        # Load and display current image
        image_path = self.image_files[self.current_image_index]
        self.current_image_path = image_path
        
        try:
            # Display main image
            image = Image.open(image_path)
            display_size = (800, 600)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Load and display label
            label_path = os.path.join(
                os.path.dirname(os.path.dirname(image_path)),
                "labels",
                os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            )
            self.current_label_path = label_path
            self._load_label(label_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def _load_label(self, label_path):
        try:
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        values = line.split()
                        if len(values) >= 5:
                            class_id = int(values[0])
                            
                            # Set class ID in entry
                            self.class_id.delete(0, tk.END)
                            self.class_id.insert(0, str(class_id))
                            
                            # Show example image for this class
                            if class_id in self.class_examples:
                                self.example_label.configure(image=self.class_examples[class_id])
                            else:
                                self.example_label.configure(image="")
                            
                            self.current_label = values
            else:
                self.class_id.delete(0, tk.END)
                self.example_label.configure(image="")
                self.current_label = None
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading label: {str(e)}")
    
    def _save_changes(self):
        if not self.current_label_path:
            return
        
        try:
            # Get class ID
            class_id_str = self.class_id.get().strip()
            if not class_id_str:
                raise ValueError("Please enter a class ID")
            
            class_id = int(class_id_str)
            
            # Keep all existing values except class ID
            if self.current_label:
                self.current_label[0] = str(class_id)
                with open(self.current_label_path, 'w') as f:
                    f.write(" ".join(self.current_label) + "\n")
                
                # Move to next image
                self._next_image()
            else:
                raise ValueError("No label data found to save")
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error saving label: {str(e)}")
    
    def _next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self._display_current_image()
    
    def _prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._display_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOLabelEditor(root)
    root.mainloop() 