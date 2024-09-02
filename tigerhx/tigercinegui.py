import tkinter as tk
from tkinter import ttk, filedialog
import os
import sys
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from guitool import *
from os.path import join, isfile, basename, dirname
from skimage.transform import resize
import numpy as np
import glob
from scipy.io import loadmat, savemat

from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion

# Determine if the application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_path = join(application_path, 'models')
output_path = join(application_path, 'output')
sample_path = join(application_path, 'csv')
csv_path = join(application_path, 'csv')
init_app(application_path)

class GlobalVariables:
    def __init__(self):
        self.log_box = None
        self.root = None
        self.progress_bar = None
        self.display_type_combo = None
        self.colormap_combo = None
        self.data = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.im = None
        self.nii_files_listbox = None
        self.seg = None
        self.norm_max = None
        self.canvas_widget = None
        self.stop_event = threading.Event()
        self.csv_path = csv_path

# Initialize the global variables
GV = GlobalVariables()

def on_go():
    global GV
    GV.stop_event.clear()

    selected_model = GV.combo.get()
    if selected_model:
        model_ff = os.path.join(model_path, selected_model)
        log_message(GV.log_box, "Processing started...")
        GV.progress_bar.pack(padx=10, pady=10, fill=tk.X)
        GV.root.update()  # Ensure the main window stays updated

        # Introduce file selection dialog
        filetypes = [("All supported files", "*.csv *.nii *.nii.gz"),
                      ("CSV files", "*.csv"),
                     ("NIfTI files", "*.nii *.nii.gz")]
        default_dir = os.path.join(application_path, 'csv')
        selected_file = filedialog.askopenfilename(initialdir=default_dir, filetypes=filetypes)

        files, options, common_path = run_program_gui_interaction(selected_file, GV.log_box, GV.root)
        
        threading.Thread(target=process_files_multithreaded,
                         args=(files, options, model_ff, common_path, GV.stop_event)).start()
    else:
        log_message(GV.log_box, "No Selection: Please select a model from the list.")

def process_files_multithreaded(files, option_list, model_ff, common_path, stop_event):
    from scipy.ndimage import zoom
  
    def resample(data, original_spacing, new_spacing, order=3):
        zoom_factors = [original_spacing[i] / new_spacing[i] for i in range(len(original_spacing))]
        resampled_data = zoom(data, zoom_factors, order=order)
        return resampled_data, zoom_factors

    def resample_back(data, zoom_factors, order=0):
        reverse_zoom_factors = [1 / zf for zf in zoom_factors]
        resampled_data = zoom(data, reverse_zoom_factors, order=order)
        return resampled_data

    onnx_version = basename(model_ff).split('_')[1]
    stopped = False
    for num, file in enumerate(files):
        option = option_list[num]
        if stop_event.is_set():
            stopped = True
            break
        if common_path is None:
            name = basename(file)
            display_name = file
        else:
            name  = os.path.relpath(file, common_path).replace(os.sep, '_')
            display_name = os.path.relpath(file, common_path)
        name = name.split('.nii')[0]
        img_ori, affine, header = load_nii(file)
        img = img_ori.copy()
        voxel_size = header.get_zooms()

        if len(img.shape) == 3:
            img = img[..., None]
            voxel_size = list(voxel_size) + [1]
        log_message(GV.log_box, f'{num + 1}/{len(files)}: Predicting {display_name} ......')

        # original 
        emp = predict_cine4d(img, model_ff, GV.progress_bar, GV.root, stop_event)

        if stop_event.is_set():
            stopped = True
            break
        
        log_message(GV.log_box, f'Selected slice for apex:  {option["Apex"]}/{img.shape[2] - 1}')
        log_message(GV.log_box, f'Creating AHA segments.........')

        LVM = emp * 0
        nseg = 6
        GV.progress_bar['value'] = 0  # Reset progress bar for inner loop
        GV.progress_bar['maximum'] = LVM.shape[2]

        for i in range(LVM.shape[2]):
            if stop_event.is_set():
                stopped = True
                break
            if i == option['Apex']:
                nseg = 4
            for j in range(LVM.shape[3]):
                if np.count_nonzero(emp[..., i, j] == 1) > 0 and np.count_nonzero(emp[..., i, j] == 2) > 0 and np.count_nonzero(emp[..., i, j] == 3) > 0:
                    LVM[..., i, j] = get_ahaseg(emp[..., i, j], nseg=nseg)
                else:
                    LVM[..., i, j] = (emp[..., i, j] == 2) * 1

            # Update the progress bar for the inner loop
            GV.progress_bar['value'] = i + 1
            GV.root.update_idletasks()  # Ensure the GUI updates

        Seg_AHA = emp.copy()
        Seg_AHA[LVM > 0] = LVM[LVM > 0] + 7
        Seg_AHA = Seg_AHA.astype(int)
        LV = (emp == 1).astype(int)
        RV = (emp == 3).astype(int)
        LVM = LVM.astype(int)
        Seg = emp.astype(int)

        mask = np.max(Seg, axis=(2, 3))
        xx, yy = np.nonzero(mask)
        x0, x1 = max(0, xx.min() - 10), min(mask.shape[0], xx.max() + 10)
        y0, y1 = max(0, yy.min() - 10), min(mask.shape[1], yy.max() + 10)

        rdict = {'input_img': img, 'Seg': Seg, 'SegAHA': Seg_AHA,
                'input_crop': img[x0:x1, y0:y1], 'Seg_crop': Seg[x0:x1, y0:y1],
                'SegAHA_crop': Seg_AHA[x0:x1, y0:y1],
                'LV': LV, 'LVM': LVM, 'RV': RV,
                'voxel_size': np.array(voxel_size)}
        
        rdict['model'] = basename(model_ff)

        if LV.shape[3] > 1: # multi-frame
            ES_t, ED_t = get_ESED(LV)
            rdict['ES_frame_0base'] = ES_t
            rdict['ED_frame_0base'] = ES_t
            rdict['Seg_ES'] = Seg[..., ES_t]
            rdict['Seg_ED'] = Seg[..., ED_t]
            rdict['SegAHA_ES'] = Seg_AHA[..., ES_t]
            rdict['SegAHA_ED'] = Seg_AHA[..., ED_t]
        savemat(f'./output/{name}_pred_{onnx_version}.mat', rdict, do_compression=True)

        if option['mat_in_inputdir']:
            mat_f = option['Filename']
            mat_f = join(dirname(mat_f), basename(mat_f).split('.')[0] + f'_{onnx_version}.mat')
            savemat(mat_f, dict, do_compression=True)

        if option['nii_in_inputdir']:
            nii_f = option['Filename']
            nii_f = join(dirname(nii_f), basename(nii_f).split('.')[0] + f'_{onnx_version}_seg.nii.gz')

            nii_img = nib.Nifti1Image(Seg, affine, header)
            nib.save(nii_img, nii_f)
        log_message(GV.log_box, f'{num + 1}/{len(files)}: {basename(file)} finished ......')
        GV.root.after(0, update_mat_listbox)

    if stopped:
        log_message(GV.log_box, f'Jobs stopped.........')
    else:
        log_message(GV.log_box, f'All job finished.........')
    GV.progress_bar['value'] = 0
    GV.root.update_idletasks()  # Ensure the GUI updates
    GV.root.after(0, update_mat_listbox)


def on_mat_select(event):
    global GV
    widget = event.widget
    selection = widget.curselection()
    if selection:
        selected_mat = widget.get(selection[0])
        if selected_mat:
            mat_path = os.path.join(output_path, selected_mat)
            try:
                GV.data = loadmat(mat_path)
                selected_type = GV.display_type_combo.get()
                on_display_type_change(0)
                log_message(GV.log_box, f"Showing {selected_mat}")
                log_message(GV.log_box, f"{selected_type} matrix size: {GV.seg.shape}")                
                
                if 'model' in GV.data:
                    model_name = GV.data['model'][0]  # from .mat file, the string stored into a cell array
                    log_message(GV.log_box, f"Predicted using {model_name}")
            except Exception as e:
                log_message(GV.log_box, f"An error occurred: {e}")


def on_display_type_change(event):
    global GV
    
    if GV.data is not None:
        GV.norm_max = np.max(GV.data['input_img'][GV.data['Seg']==1])
        if GV.norm_max == 0:
            GV.norm_max = np.max(GV.data['input_img'])
        selected_type = GV.display_type_combo.get()

        if selected_type == 'edge':
            GV.seg = get_edge(GV.data['input_img'], GV.data['Seg'], GV.norm_max)
        elif selected_type == 'edge_crop':
            GV.seg = get_edge(GV.data['input_crop'], GV.data['Seg_crop'], GV.norm_max)
        else:                                  
            GV.seg = GV.data[selected_type]
        show_montage(GV.seg)
        update_time_slider(GV.seg)  # Adapt the range of the time points
    try:
        pass
    except:
        log_message(GV.log_box, f"Select a correct result file....")

def on_colormap_change(event):
    global GV
    if GV.seg is not None:
        show_montage(GV.seg)  # Redraw the figure with the selected colormap

def show_montage(emp, time_frame=0):
    global GV
    plt.close('all')  # Close all previous figures

    # Resize the mosaic to 400x600
    height = 300
    width = 600

    # Initialize the figure and axes if they do not exist
    if GV.fig is None or GV.ax is None:
        GV.fig, GV.ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        GV.ax.axis('off')  # Hide the axes
        GV.ax = GV.fig.add_axes([0, 0, 1, 1])  # Remove all margins and padding
        GV.ax.set_facecolor('black')
        GV.fig.set_facecolor('black')

    # Clear previous canvas content
    for widget in GV.canvas_frame.winfo_children():
        widget.destroy()

    cmap = GV.colormap_combo.get()  # Get the selected colormap

    # Create the padded mosaic
    padded_mosaic = create_padded_mosaic(emp, time_frame, height / width)
    selected_type = GV.display_type_combo.get()
    if 'edge' in selected_type:
        newmosaic = padded_mosaic.copy()
        edge = (newmosaic == 255)
        newmosaic[edge] = 0

        mosaic_resized = resize(newmosaic, (height, width), order=0, preserve_range=True).astype(int)
        edge = resize(edge, (height, width), order=0, preserve_range=True)
        mosaic_resized[edge] = 255

        base_cmap = plt.get_cmap(cmap)
        base_colors = base_cmap(np.arange(256))
        base_colors[255] = (1, 0, 0, 1)
        cmap = ListedColormap(base_colors)
        display_min = 0
        display_max = 255
    else:
        mosaic_resized = resize(padded_mosaic, (height, width))
        display_min = emp.min()
        display_max = GV.norm_max

    if ('Seg' in selected_type) or (selected_type in ['RV', 'LV', 'LVM']):
        display_min = emp.min()
        display_max = emp.max()

    if GV.im is None:
        GV.im = GV.ax.imshow(mosaic_resized, cmap=cmap, interpolation='nearest')
        GV.im.set_clim(vmin=display_min, vmax=display_max)
    else:
        GV.im.set_data(mosaic_resized)
        GV.im.set_cmap(cmap)
        GV.im.set_clim(vmin=display_min, vmax=display_max)

    GV.canvas = FigureCanvasTkAgg(GV.fig, master=GV.canvas_frame)
    GV.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
    GV.canvas.draw()


def update_montage(time_frame):
    global GV
    show_montage(GV.seg, time_frame)

def update_time_slider(emp):
    global GV
    if len(emp.shape) == 3: emp = emp[..., None]
    max_time_frame = emp.shape[3] - 1  # Get the maximum time frame
    GV.time_slider.config(to=max_time_frame)  # Update the slider's range
    GV.time_slider.set(0)  # Set initial value to 0

def update_mat_listbox():
    global GV
    mat_files = list_mat_files(output_path)
    GV.mat_listbox.delete(0, tk.END)
    for file in mat_files:
        GV.mat_listbox.insert(tk.END, file)

def on_closing():
    global GV
    plt.close('all')  # Close all matplotlib figures
    GV.root.destroy()
    sys.exit()  # Ensure the program ends

def on_select_folder():
    global GV
    select_folder(GV)

def stop_processing():
    global GV
    log_message(GV.log_box, "Processing stopped by user.")
    GV.stop_event.set()

# Create the main window
GV.root = tk.Tk()
GV.root.title("TigerHx GUI (NTUST X NTHU)")

# Set default font size
default_font = ("Arial", 10)
GV.root.option_add("*Font", default_font)

# Adjust the size of the main window
screen_width = GV.root.winfo_screenwidth()
screen_height = GV.root.winfo_screenheight()
window_width = 1100
window_height = 600
GV.root.geometry(f"{window_width}x{window_height}")

# Handle window close event
GV.root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a frame for the combo box and label
frame = tk.Frame(GV.root)
frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)

# Create a frame for the combo box and "Go" button
combo_frame = tk.Frame(frame)
combo_frame.pack(pady=5)

# Create a label for the combo box
label = tk.Label(combo_frame, text="Model")
label.pack(side=tk.LEFT, pady=5)

onnx_files = list_onnx_files(model_path)

# 创建ComboBox并设置默认选择最后一个ONNX文件
GV.combo = ttk.Combobox(combo_frame, values=onnx_files, width=30)
GV.combo.pack(side=tk.LEFT, padx=5)
GV.combo.current(len(onnx_files) - 1)


# Create a frame for the combo box and "Go" button
button_frame = tk.Frame(frame)
button_frame.pack(pady=5)

# Create a button to select a folder
select_folder_button = tk.Button(button_frame, text="GenCSV", command=on_select_folder)
select_folder_button.pack(side=tk.LEFT, padx=5)

# Create a "Go" button
go_button = tk.Button(button_frame, text="Select CSV or NII and RUN", command=on_go)
go_button.pack(side=tk.LEFT, padx=5)

# Create a "Stop" button
stop_button = tk.Button(button_frame, text="Stop", command=stop_processing)
stop_button.pack(side=tk.LEFT, padx=5)

# Create a log box to display messages
log_frame = tk.Frame(frame)
log_frame.pack(padx=10, pady=10)

GV.log_box = tk.Text(log_frame, width=50, height=10, wrap=tk.WORD)
GV.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

log_scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=GV.log_box.yview)
log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
GV.log_box.config(yscrollcommand=log_scrollbar.set)

# Create a frame for the listbox and display type combo box
listbox_frame = tk.Frame(frame)
listbox_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a frame for the display type label and combo box
display_type_frame = tk.Frame(listbox_frame)
display_type_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

# Create a label for the display type combo box
display_type_label = tk.Label(display_type_frame, text="Figure type")
display_type_label.pack(side=tk.LEFT)

# Create a combo box for selecting display type
display_types = ['input_img', 'edge', 'Seg', 'SegAHA',
                 'input_crop', 'edge_crop', 'Seg_crop',
                 'SegAHA_crop', 'LV', 'LVM', 'RV']
GV.display_type_combo = ttk.Combobox(display_type_frame, values=display_types, width=16)
GV.display_type_combo.pack(side=tk.LEFT, padx=5)
GV.display_type_combo.current(0)  # Set default display type to 'Seg'
GV.display_type_combo.bind("<<ComboboxSelected>>", on_display_type_change)

# Create a label for the colormap combo box
colormap_label = tk.Label(display_type_frame, text="Colormap")
colormap_label.pack(side=tk.LEFT, padx=5)

# Create a combo box for selecting colormap
GV.colormap_combo = ttk.Combobox(display_type_frame, values=['gray', 'viridis'], width=10)
GV.colormap_combo.pack(side=tk.LEFT, padx=5)
GV.colormap_combo.current(0)  # Set default colormap to 'gray'
GV.colormap_combo.bind("<<ComboboxSelected>>", on_colormap_change)

# Create a scrollbar for the listbox
listbox_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a listbox for the .mat files
GV.mat_listbox = tk.Listbox(listbox_frame, width=40, height=10, yscrollcommand=listbox_scrollbar.set)
GV.mat_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
GV.mat_listbox.bind('<<ListboxSelect>>', on_mat_select)

# Configure the scrollbar to work with the listbox
listbox_scrollbar.config(command=GV.mat_listbox.yview)

# Create a progress bar
GV.progress_bar = ttk.Progressbar(frame, mode='determinate')
GV.progress_bar.pack(side=tk.TOP, fill=tk.BOTH)

# Create a frame for the canvas and slider
canvas_slider_frame = tk.Frame(GV.root, width=650, height=450)
canvas_slider_frame.pack(padx=2, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=False)

# Create a frame for the canvas
GV.canvas_frame = tk.Frame(canvas_slider_frame, width=600, height=300, bg='black')  # Set background to black and fix size
GV.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

# Create the time slider
GV.time_slider = tk.Scale(canvas_slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=lambda val: update_montage(int(val)))
GV.time_slider.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

# Initial update of the .mat listbox
update_mat_listbox()

welcome_msg = f'''\n* Step 1: Click GenCSV to search Cine NIFTI files.
* Step 2: TigerHx will generate a CSV file in {sample_path}.
* Step 3: Edit the CSV file and assign APEX numbers.
* Step 4: Click 'RUN' to start the automatic segmentation.
* Step 5: Click the prediction files to inspect the results.'''
log_message(GV.log_box, welcome_msg)

welcome_msg = f'''\n* 1: 點選GenCSV去搜尋 Cine NIfTI 資料集
* 2: TigerHx 將在 {sample_path} 生成一個 CSV 檔案
* 3: 編輯 CSV 檔案並分配 APEX 編號
* 4: 點擊 RUN 開始使用 TigerHx 進行自動心臟分割
* 5: 點擊預測檔案以檢查結果\n'''
log_message(GV.log_box, welcome_msg)

# Run the application
GV.root.mainloop()
