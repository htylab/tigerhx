import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tigerhx import guitool
from os.path import join, isfile, basename, dirname
from skimage.transform import resize
import numpy as np
import glob
from scipy.io import loadmat, savemat
from matplotlib.colors import ListedColormap

# Determine if the application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_path = join(application_path, 'models')
output_path = join(application_path, 'output')
sample_path = join(application_path, 'csv')
csv_path = join(application_path, 'csv')
guitool.init_app(application_path)

# --- Theme & Style Settings ---
BG_COLOR = "#2e2e2e"
FG_COLOR = "#ffffff"
ACCENT_COLOR = "#007acc"  # A nice blue
BUTTON_BG = "#3e3e3e"
BUTTON_FG = "#ffffff"
TEXT_BG = "#1e1e1e"
TEXT_FG = "#dcdcdc"
FONT_MAIN = ("Segoe UI", 10)
FONT_HEADER = ("Segoe UI", 12, "bold")

class TigerCineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TigerHx GUI v2 (Modernized)")
        self.geometry("1200x700")
        self.configure(bg=BG_COLOR)
        
        # Set window icon if available
        # try: self.iconbitmap(join(application_path, 'tigerhx.ico'))
        # except: pass

        self.setup_styles()
        self.init_variables()
        self.create_layout()
        self.update_mat_listbox()
        
        # Initial Welcome Message
        welcome_msg = (
            "Welcome to TigerHx v2!\n\n"
            "1. Click 'GenCSV' to scan for Cine NIfTI files.\n"
            "2. Edit the generated CSV in the 'csv' folder if needed.\n"
            "3. Select a Model and Click 'RUN' to start segmentation.\n"
            "4. Select output files from the list to visualize results."
        )
        self.log_message(welcome_msg)

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')  # 'clam' is usually a good base for custom coloring

        # Configure generic TFrame
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabelframe", background=BG_COLOR, foreground=FG_COLOR)
        style.configure("TLabelframe.Label", background=BG_COLOR, foreground=FG_COLOR, font=FONT_HEADER)
        
        # Configure TLabel
        style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=FONT_MAIN)
        
        # Configure TButton
        style.configure("TButton", 
                        background=BUTTON_BG, 
                        foreground=BUTTON_FG, 
                        borderwidth=1, 
                        focusthickness=3, 
                        focuscolor=ACCENT_COLOR,
                        font=FONT_MAIN)
        style.map("TButton", 
                  background=[('active', ACCENT_COLOR)], 
                  foreground=[('active', 'white')])

        # Configure TCombobox
        style.configure("TCombobox", fieldbackground=TEXT_BG, background=BUTTON_BG, foreground=TEXT_FG, arrowcolor=FG_COLOR)
        style.map("TCombobox", fieldbackground=[('readonly', TEXT_BG)])

        # Configure TProgressbar
        style.configure("Horizontal.TProgressbar", background=ACCENT_COLOR, troughcolor=TEXT_BG, bordercolor=BG_COLOR)

        # Configure TScale
        style.configure("Horizontal.TScale", background=BG_COLOR, troughcolor=TEXT_BG, sliderthickness=15)

    def init_variables(self):
        self.stop_event = threading.Event()
        self.csv_path = csv_path
        self.data = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.im = None
        self.seg = None
        self.norm_max = None
        
        # These will be bound to widgets later
        self.selected_model = tk.StringVar()
        self.selected_display_type = tk.StringVar(value='Seg')
        self.selected_colormap = tk.StringVar(value='gray')

    def create_layout(self):
        # --- Main Container ---
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Panel: Controls & Log ---
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Control Group
        control_group = ttk.LabelFrame(left_panel, text="Controls", padding=10)
        control_group.pack(fill=tk.X, pady=(0, 10))

        # Model Selection
        ttk.Label(control_group, text="Model:").pack(anchor='w')
        onnx_files = guitool.list_onnx_files(model_path)
        self.model_combo = ttk.Combobox(control_group, textvariable=self.selected_model, values=onnx_files)
        if onnx_files: self.model_combo.current(len(onnx_files) - 1)
        self.model_combo.pack(fill=tk.X, pady=(0, 10))

        # Action Buttons
        btn_frame = ttk.Frame(control_group)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="GenCSV", command=self.on_gen_csv).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="RUN", command=self.on_run).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Stop", command=self.on_stop).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # Progress Bar
        self.progress_bar = ttk.Progressbar(control_group, mode='determinate', style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))

        # Log Area
        log_group = ttk.LabelFrame(left_panel, text="Log", padding=10)
        log_group.pack(fill=tk.BOTH, expand=True)
        
        self.log_box = tk.Text(log_group, bg=TEXT_BG, fg=TEXT_FG, insertbackground=FG_COLOR, 
                               relief=tk.FLAT, font=("Consolas", 9), height=15)
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_group, orient=tk.VERTICAL, command=self.log_box.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_box.config(yscrollcommand=scrollbar.set)

        # --- Right Panel: Visualization & Results ---
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Visualization Area (Top Right)
        viz_group = ttk.LabelFrame(right_panel, text="Visualization", padding=10)
        viz_group.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))

        # Canvas Frame (Matplotlib)
        self.canvas_frame = tk.Frame(viz_group, bg="black")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Time Slider
        self.time_slider = ttk.Scale(viz_group, from_=0, to=0, orient=tk.HORIZONTAL, 
                                     command=lambda val: self.update_montage(int(float(val))))
        self.time_slider.pack(fill=tk.X, pady=(10, 0))

        # Bottom Right: Result List & Display Settings
        bottom_right = ttk.Frame(right_panel, height=200)
        bottom_right.pack(side=tk.BOTTOM, fill=tk.X)

        # Result List
        list_group = ttk.LabelFrame(bottom_right, text="Results", padding=10)
        list_group.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.result_listbox = tk.Listbox(list_group, bg=TEXT_BG, fg=TEXT_FG, selectbackground=ACCENT_COLOR,
                                         relief=tk.FLAT, font=FONT_MAIN)
        self.result_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_listbox.bind('<<ListboxSelect>>', self.on_mat_select)
        
        list_scroll = ttk.Scrollbar(list_group, orient=tk.VERTICAL, command=self.result_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_listbox.config(yscrollcommand=list_scroll.set)

        # Display Settings
        settings_group = ttk.LabelFrame(bottom_right, text="View Settings", padding=10)
        settings_group.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(settings_group, text="Type:").pack(anchor='w')
        display_types = ['input_img', 'edge', 'Seg', 'SegAHA', 'input_crop', 'edge_crop', 'Seg_crop', 'SegAHA_crop', 'LV', 'LVM', 'RV']
        self.type_combo = ttk.Combobox(settings_group, textvariable=self.selected_display_type, values=display_types)
        self.type_combo.pack(fill=tk.X, pady=(0, 10))
        self.type_combo.bind("<<ComboboxSelected>>", self.on_display_type_change)

        ttk.Label(settings_group, text="Colormap:").pack(anchor='w')
        self.cmap_combo = ttk.Combobox(settings_group, textvariable=self.selected_colormap, values=['gray', 'viridis', 'magma', 'plasma'])
        self.cmap_combo.pack(fill=tk.X)
        self.cmap_combo.bind("<<ComboboxSelected>>", self.on_colormap_change)

    # --- Logic Methods ---

    def log_message(self, message):
        self.log_box.insert(tk.END, message + '\n')
        self.log_box.see(tk.END)

    def on_gen_csv(self):
        # We need to bridge to guitool.select_folder, but it expects a 'GV' object with 'log_box', 'csv_path', 'root'
        # Let's create a temporary proxy object
        class GVProxy:
            def __init__(proxy_self, parent):
                proxy_self.log_box = parent.log_box
                proxy_self.csv_path = parent.csv_path
                proxy_self.root = parent
        
        guitool.select_folder(GVProxy(self))

    def on_run(self):
        self.stop_event.clear()
        selected_model = self.selected_model.get()
        if not selected_model:
            self.log_message("Error: Please select a model.")
            return

        model_ff = os.path.join(model_path, selected_model)
        self.log_message("--- Processing Started ---")
        
        # File Selection
        filetypes = [("All supported files", "*.csv *.nii *.nii.gz"), ("CSV files", "*.csv"), ("NIfTI files", "*.nii *.nii.gz")]
        default_dir = os.path.join(application_path, 'csv')
        selected_file = filedialog.askopenfilename(initialdir=default_dir, filetypes=filetypes)

        if not selected_file:
            self.log_message("No file selected.")
            return

        files, options, common_path = guitool.run_program_gui_interaction(selected_file, self.log_box, self)
        
        if files:
             threading.Thread(target=self.process_files_multithreaded,
                             args=(files, options, model_ff, common_path)).start()

    def on_stop(self):
        self.stop_event.set()
        self.log_message("Stopping processing...")

    def process_files_multithreaded(self, files, option_list, model_ff, common_path):
        # This is a port/adaptation of the logic in the original file
        from scipy.ndimage import zoom
        
        onnx_version = basename(model_ff).split('_')[1]
        stopped = False
        
        # We need to attach progress bar to something accessible or update it here.
        # Original code updated GV.progress_bar directly.
        
        for num, file in enumerate(files):
            if self.stop_event.is_set():
                stopped = True
                break
            
            option = option_list[num]
            if common_path is None:
                name = basename(file)
                display_name = file
            else:
                name  = os.path.relpath(file, common_path).replace(os.sep, '_')
                display_name = os.path.relpath(file, common_path)
            
            name = name.split('.nii')[0]
            
            # Load NIfTI
            try:
                img_ori, affine, header = guitool.load_nii(file)
            except Exception as e:
                self.log_message(f"Error loading {file}: {e}")
                continue

            img = img_ori.copy()
            voxel_size = header.get_zooms()

            if len(img.shape) == 3:
                img = img[..., None]
                voxel_size = list(voxel_size) + [1]
            
            self.log_message(f'{num + 1}/{len(files)}: Predicting {display_name} ...')
            
            # --- Prediction ---
            # Pass a mock progress bar if needed, or handle progress updates manually
            # The guitool.predict_cine4d function takes 'progress_bar' and 'root'.
            # We can pass self.progress_bar and self (as root).
            
            self.progress_bar['value'] = 0
            emp = guitool.predict_cine4d(img, model_ff, self.progress_bar, self, self.stop_event)

            if self.stop_event.is_set():
                stopped = True
                break

            # --- Post Processing (AHA Segments) ---
            self.log_message(f'Creating AHA segments (Apex: {option["Apex"]})...')
            
            LVM = emp * 0
            nseg = 6
            self.progress_bar['maximum'] = LVM.shape[2]
            
            for i in range(LVM.shape[2]):
                if self.stop_event.is_set():
                    stopped = True
                    break
                if i == option['Apex']:
                    nseg = 4
                
                # Slices loop
                for j in range(LVM.shape[3]):
                    slice_mask = emp[..., i, j]
                    if (np.any(slice_mask == 1) and np.any(slice_mask == 2) and np.any(slice_mask == 3)):
                        LVM[..., i, j] = guitool.get_ahaseg(slice_mask, nseg=nseg)
                    else:
                        LVM[..., i, j] = (slice_mask == 2) * 1
                
                self.progress_bar['value'] = i + 1
                self.update_idletasks()

            # --- Saving Results ---
            Seg_AHA = emp.copy()
            Seg_AHA[LVM > 0] = LVM[LVM > 0] + 7
            Seg_AHA = Seg_AHA.astype(int)
            LV = (emp == 1).astype(int)
            RV = (emp == 3).astype(int)
            LVM = LVM.astype(int)
            Seg = emp.astype(int)

            # Cropping for display/save
            mask = np.max(Seg, axis=(2, 3))
            if np.any(mask):
                xx, yy = np.nonzero(mask)
                x0, x1 = max(0, xx.min() - 10), min(mask.shape[0], xx.max() + 10)
                y0, y1 = max(0, yy.min() - 10), min(mask.shape[1], yy.max() + 10)
            else:
                x0, x1, y0, y1 = 0, mask.shape[0], 0, mask.shape[1]

            rdict = {
                'input_img': img, 'Seg': Seg, 'SegAHA': Seg_AHA,
                'input_crop': img[x0:x1, y0:y1], 'Seg_crop': Seg[x0:x1, y0:y1],
                'SegAHA_crop': Seg_AHA[x0:x1, y0:y1],
                'LV': LV, 'LVM': LVM, 'RV': RV,
                'voxel_size': np.array(voxel_size),
                'model': basename(model_ff)
            }

            if LV.shape[3] > 1:
                ES_t, ED_t = guitool.get_ESED(LV)
                rdict.update({
                    'ES_frame_0base': ES_t, 'ED_frame_0base': ED_t,
                    'Seg_ES': Seg[..., ES_t], 'Seg_ED': Seg[..., ED_t],
                    'SegAHA_ES': Seg_AHA[..., ES_t], 'SegAHA_ED': Seg_AHA[..., ED_t]
                })
            
            savemat(f'./output/{name}_pred_{onnx_version}.mat', rdict, do_compression=True)

            # Optional extra saves
            if option.get('mat_in_inputdir'):
                mat_f = join(dirname(option['Filename']), basename(option['Filename']).split('.')[0] + f'_{onnx_version}.mat')
                savemat(mat_f, rdict, do_compression=True)

            if option.get('nii_in_inputdir'):
                nii_f = join(dirname(option['Filename']), basename(option['Filename']).split('.')[0] + f'_{onnx_version}_seg.nii.gz')
                nii_img = nib.Nifti1Image(Seg, affine, header)
                nib.save(nii_img, nii_f)

            self.log_message(f'{basename(file)} finished.')
            self.after(0, self.update_mat_listbox)

        if stopped:
            self.log_message('Jobs stopped.')
        else:
            self.log_message('All jobs finished.')
        
        self.progress_bar['value'] = 0

    def update_mat_listbox(self):
        mat_files = guitool.list_mat_files(output_path)
        self.result_listbox.delete(0, tk.END)
        for file in mat_files:
            self.result_listbox.insert(tk.END, file)

    def on_mat_select(self, event):
        selection = self.result_listbox.curselection()
        if selection:
            selected_mat = self.result_listbox.get(selection[0])
            mat_path = os.path.join(output_path, selected_mat)
            try:
                self.data = loadmat(mat_path)
                self.log_message(f"Loaded {selected_mat}")
                if 'model' in self.data:
                    # Handle cell array vs string
                    m = self.data['model']
                    model_name = m[0] if isinstance(m, (list, np.ndarray)) and len(m) > 0 else m
                    self.log_message(f"Model: {model_name}")
                
                self.time_slider.set(0) # Reset time to 0 to avoid index errors on new file
                self.on_display_type_change() # Trigger display update
            except Exception as e:
                self.log_message(f"Error loading MAT: {e}")

    def on_display_type_change(self, event=None):
        if self.data is None: return
        
        # Calculate normalization max if not set
        if self.norm_max is None or True: # Always recalculate for safety on new file
            seg_mask = self.data['Seg'] == 1
            if np.any(seg_mask):
                self.norm_max = np.max(self.data['input_img'][seg_mask])
            else:
                self.norm_max = np.max(self.data['input_img'])
                if self.norm_max == 0: self.norm_max = 1

        selected_type = self.selected_display_type.get()
        
        if selected_type == 'edge':
            self.seg = guitool.get_edge(self.data['input_img'], self.data['Seg'], self.norm_max)
        elif selected_type == 'edge_crop':
            self.seg = guitool.get_edge(self.data['input_crop'], self.data['Seg_crop'], self.norm_max)
        else:
             if selected_type in self.data:
                 self.seg = self.data[selected_type]
             else:
                 self.log_message(f"Type {selected_type} not found in data.")
                 return

        self.update_time_slider(self.seg)
        
        # Validate time frame
        current_time = int(self.time_slider.get())
        if len(self.seg.shape) > 3:
            max_t = self.seg.shape[3] - 1
            if current_time > max_t:
                current_time = 0
                self.time_slider.set(0)
        
        self.show_montage(self.seg, current_time)

    def on_colormap_change(self, event=None):
        if self.seg is not None:
            self.show_montage(self.seg, self.time_slider.get())

    def update_time_slider(self, emp):
        if len(emp.shape) == 3: emp = emp[..., None]
        max_time_frame = emp.shape[3] - 1
        self.time_slider.config(to=max_time_frame)
        # self.time_slider.set(0) # Keep current pos if possible?
    
    def update_montage(self, time_frame):
        if self.seg is not None:
            self.show_montage(self.seg, time_frame)

    def show_montage(self, emp, time_frame=0):
        # Clean up
        if self.fig is None:
            self.create_figure()
        
        try:
            cmap_name = self.selected_colormap.get()
            
            # Prepare Data
            padded_mosaic = guitool.create_padded_mosaic(emp, time_frame, aspect_ratio=0.66)
            
            selected_type = self.selected_display_type.get()
            
            display_min = 0
            display_max = self.norm_max
            
            if 'edge' in selected_type:
                 display_max = 255
            elif ('Seg' in selected_type) or (selected_type in ['RV', 'LV', 'LVM']):
                display_min = emp.min()
                display_max = emp.max()

            self.ax.clear()
            self.ax.axis('off')
            
            # Special colormap handling for edges if needed, otherwise standard
            if 'edge' in selected_type and cmap_name == 'gray':
                 # Maybe make edges red?
                 self.im = self.ax.imshow(padded_mosaic, cmap='gray', vmin=0, vmax=255)
            else:
                 self.im = self.ax.imshow(padded_mosaic, cmap=cmap_name, vmin=display_min, vmax=display_max, interpolation='nearest')
            
            self.canvas.draw()
        except Exception as e:
            self.log_message(f"Error displaying montage: {e}")

    def create_figure(self):
        # 600x400 pixels approx
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor('black')
        self.ax.axis('off')
        self.ax.set_facecolor('black')
        
        # Remove margins
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = TigerCineGUI()
    app.mainloop()
