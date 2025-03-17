import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
import re
from PIL import Image, ImageTk
from collections import defaultdict

class ReferenceWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Label References")
        self.geometry("800x600")
        
        # Create a canvas with scrollbar
        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Dictionary to store label images
        self.label_images = {}  # Keep references to avoid garbage collection
        
    def update_references(self, label_images):
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Clear stored images
        self.label_images.clear()
        
        # Add new reference images
        for label, image_path in sorted(label_images.items()):
            frame = ttk.LabelFrame(self.scrollable_frame, text=f"Label {label}", padding="5")
            frame.pack(fill="x", padx=5, pady=5)
            
            try:
                image = Image.open(image_path)
                # Resize image to a reasonable size (e.g., 200x200)
                image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                # Store reference to avoid garbage collection
                self.label_images[label] = photo
                
                label_widget = ttk.Label(frame, image=photo)
                label_widget.pack(padx=5, pady=5)
                
                # Add filename
                filename_label = ttk.Label(frame, text=os.path.basename(image_path))
                filename_label.pack(padx=5, pady=2)
            except Exception as e:
                error_label = ttk.Label(frame, text=f"Error loading image: {str(e)}")
                error_label.pack(padx=5, pady=5)

class ImageRenamerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image File Renamer")
        self.root.geometry("1200x800")
        
        # Variables
        self.folder_path = tk.StringVar()
        self.pattern = tk.StringVar(value="image_{label}_{index}")
        self.start_number = tk.StringVar(value="1")
        self.current_label = tk.StringVar(value="1")
        self.current_filename = tk.StringVar(value="")
        self.current_sequence = tk.StringVar(value="")
        self.preview_data = []
        self.current_image_index = -1
        self.image_files = []
        self.reference_window = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Main container
        main_container = ttk.PanedWindow(self.root, orient="horizontal")
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=40)
        
        # Folder selection
        folder_frame = ttk.LabelFrame(left_panel, text="Folder Selection", padding="10")
        folder_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(folder_frame, textvariable=self.folder_path, width=50).pack(side="left", padx=5)
        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).pack(side="left")
        
        # Pattern configuration
        pattern_frame = ttk.LabelFrame(left_panel, text="Rename Pattern", padding="10")
        pattern_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(pattern_frame, text="Pattern:").pack(side="left")
        ttk.Entry(pattern_frame, textvariable=self.pattern, width=30).pack(side="left", padx=5)
        ttk.Label(pattern_frame, text="Start Number:").pack(side="left")
        ttk.Entry(pattern_frame, textvariable=self.start_number, width=10).pack(side="left", padx=5)
        
        # Label configuration
        label_frame = ttk.LabelFrame(left_panel, text="Image Label", padding="10")
        label_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(label_frame, text="Label Number:").pack(side="left")
        ttk.Entry(label_frame, textvariable=self.current_label, width=10).pack(side="left", padx=5)
        ttk.Button(label_frame, text="Apply Label", command=self._apply_current_label).pack(side="left", padx=5)
        ttk.Button(label_frame, text="Show References", command=self._toggle_reference_window).pack(side="right", padx=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(left_panel)
        nav_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self._prev_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Next", command=self._next_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Preview Changes", command=self._preview_changes).pack(side="right", padx=5)
        
        # Preview area
        preview_frame = ttk.LabelFrame(left_panel, text="Rename Preview", padding="10")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview
        self.tree = ttk.Treeview(preview_frame, columns=("Original", "New", "Label"), show="headings")
        self.tree.heading("Original", text="Original Name")
        self.tree.heading("New", text="New Name")
        self.tree.heading("Label", text="Label")
        self.tree.pack(fill="both", expand=True)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Right panel for image preview
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=60)
        
        # Image info frame
        info_frame = ttk.Frame(right_panel)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # Sequence number display
        sequence_label = ttk.Label(info_frame, text="Image:")
        sequence_label.pack(side="left", padx=5)
        ttk.Label(info_frame, textvariable=self.current_sequence).pack(side="left", padx=5)
        
        # Filename display
        filename_label = ttk.Label(info_frame, text="Filename:")
        filename_label.pack(side="left", padx=(20, 5))
        ttk.Label(info_frame, textvariable=self.current_filename).pack(side="left", padx=5)
        
        # Image preview
        self.image_label = ttk.Label(right_panel)
        self.image_label.pack(fill="both", expand=True)
        
        # Keyboard bindings for navigation
        self.root.bind('<Left>', lambda e: self._prev_image())
        self.root.bind('<Right>', lambda e: self._next_image())
        self.root.bind('<Return>', lambda e: self._apply_current_label())
        
        # Bottom buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(button_frame, text="Rename Files", command=self._rename_files).pack(side="right", padx=5)
    
    def _browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
            self._load_images()
            self._preview_changes()
    
    def _load_images(self):
        folder = self.folder_path.get()
        if not folder:
            return
            
        # Get all image files and sort them naturally
        self.image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.image_files.sort(key=lambda x: self._natural_sort_key(x))
        self.current_image_index = 0 if self.image_files else -1
        self._display_current_image()
    
    def _display_current_image(self):
        if self.current_image_index < 0 or not self.image_files:
            self.image_label.configure(image='')
            self.current_filename.set("")
            self.current_sequence.set("")
            return
            
        try:
            current_file = self.image_files[self.current_image_index]
            image_path = os.path.join(self.folder_path.get(), current_file)
            image = Image.open(image_path)
            
            # Update sequence and filename display
            self.current_sequence.set(f"{self.current_image_index + 1}/{len(self.image_files)}")
            self.current_filename.set(current_file)
            
            # Calculate resize ratio to fit in 800x600 box
            display_size = (800, 600)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Highlight current item in treeview
            for item in self.tree.get_children():
                if self.tree.item(item)['values'][0] == current_file:
                    self.tree.selection_set(item)
                    self.tree.see(item)
                    break
                    
            # Update window title with current image info
            self.root.title(f"Image File Renamer - {self.current_sequence.get()} - {current_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def _next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self._display_current_image()
    
    def _prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._display_current_image()
    
    def _toggle_reference_window(self):
        if self.reference_window is None or not self.reference_window.winfo_exists():
            self.reference_window = ReferenceWindow(self.root)
            self._update_reference_images()
        else:
            self.reference_window.destroy()
            self.reference_window = None
            
    def _update_reference_images(self):
        if not self.reference_window or not self.preview_data:
            return
            
        # Create a dictionary to store one image path per label
        label_images = {}
        folder = self.folder_path.get()
        
        # Group files by label
        for old_name, _, label in self.preview_data:
            if label not in label_images:
                label_images[label] = os.path.join(folder, old_name)
        
        # Update the reference window
        self.reference_window.update_references(label_images)
    
    def _apply_current_label(self):
        if self.current_image_index < 0 or not self.image_files:
            return
            
        try:
            label = int(self.current_label.get())
            current_file = self.image_files[self.current_image_index]
            
            # Update preview data with new label
            for i, (old_name, new_name, old_label) in enumerate(self.preview_data):
                if old_name == current_file:
                    start_num = int(self.start_number.get())
                    ext = os.path.splitext(old_name)[1]
                    new_name = self.pattern.get().format(label=label, index=start_num + i) + ext
                    self.preview_data[i] = (old_name, new_name, label)
                    
                    # Update treeview
                    for item in self.tree.get_children():
                        if self.tree.item(item)['values'][0] == old_name:
                            self.tree.item(item, values=(old_name, new_name, label))
                            break
                    break
            
            # Update reference window
            self._update_reference_images()
            
            # Move to next image
            self._next_image()
        except ValueError:
            messagebox.showerror("Error", "Label must be a number!")
    
    def _preview_changes(self):
        folder = self.folder_path.get()
        if not folder:
            messagebox.showwarning("Warning", "Please select a folder first!")
            return
            
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        self.preview_data = []
        try:
            start_num = int(self.start_number.get())
        except ValueError:
            messagebox.showerror("Error", "Start number must be an integer!")
            return
            
        # Initialize preview data with default label 1
        for idx, old_name in enumerate(self.image_files):
            ext = os.path.splitext(old_name)[1]
            new_name = self.pattern.get().format(label=1, index=start_num + idx) + ext
            self.preview_data.append((old_name, new_name, 1))
            self.tree.insert("", "end", values=(old_name, new_name, 1))
    
    def _rename_files(self):
        if not self.preview_data:
            messagebox.showwarning("Warning", "Please preview changes first!")
            return
            
        folder = self.folder_path.get()
        success_count = 0
        error_count = 0
        
        for old_name, new_name, _ in self.preview_data:
            try:
                os.rename(
                    os.path.join(folder, old_name),
                    os.path.join(folder, new_name)
                )
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error renaming {old_name}: {str(e)}")
        
        messagebox.showinfo(
            "Complete",
            f"Renaming complete!\nSuccessful: {success_count}\nFailed: {error_count}"
        )
        self._load_images()  # Reload images after renaming
        self._preview_changes()

    def _natural_sort_key(self, s):
        # Helper function for natural sorting of filenames
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRenamerGUI(root)
    root.mainloop()
