import tkinter as tk
from tkinter import ttk, filedialog
import os
import sys
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from guitool import *
from os.path import join, isfile, basename
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
init_app(application_path)

# Global variables
log_box = None
root = None
progress_bar = None
display_type_combo = None
colormap_combo = None
data = None  # Ensure data is globally accessible
fig, ax = None, None
canvas = None
im = None
nii_files_listbox = None
seg = None
norm_max = None
canvas_widget = None
stop_event = threading.Event()

def on_go():
    global progress_bar, stop_event
    stop_event.clear()

    selected_model = combo.get()
    if selected_model:
        model_ff = os.path.join(model_path, selected_model)
        log_message(log_box, "Processing started...")
        progress_bar.pack(padx=10, pady=10, fill=tk.X)
        root.update()  # Ensure the main window stays updated

        # Introduce file selection dialog
        filetypes = [("All supported files", "*.csv *.nii *.nii.gz"),
                      ("CSV files", "*.csv"),
                     ("NIfTI files", "*.nii *.nii.gz")]
        default_dir = os.path.join(application_path, 'csv')
        selected_file = filedialog.askopenfilename(initialdir=default_dir, filetypes=filetypes)


        files, slice_select, common_path = run_program_gui_interaction(selected_file, log_box, root)

        threading.Thread(target=process_files_multithreaded,
                         args=(files, slice_select, model_ff, common_path, stop_event)).start()
    else:
        log_message(log_box, "No Selection: Please select a model from the list.")

def process_files_multithreaded(files, slice_select, model_ff, common_path, stop_event):
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
        if stop_event.is_set():
            stopped = True
            break
        if common_path is None:
            name = basename(file)
        else:
            name  = os.path.relpath(file, common_path).replace(os.sep, '_')
        name = name.split('.nii')[0]
        img_ori, affine, header = load_nii(file)
        img = img_ori.copy()
        voxel_size = header.get_zooms()

        if len(img.shape) == 3:
            img = img[..., None]
            voxel_size = list(voxel_size) + [1]
        log_message(log_box, f'{num + 1}/{len(files)}: Predicting {basename(file)} ......')


        # 調整解析度至(1, 1, original_spacing[2], original_spacing[3])
        #new_spacing = (1, 1, voxel_size[2], voxel_size[3])
        #resampled_img, zoom_factors = resample(img, voxel_size, new_spacing, order=3)
        #mask = predict_cine4d(resampled_img, model_ff, progress_bar, root, stop_event)
        #emp = resample_back(mask, zoom_factors, order=0)


        #original 
        emp = predict_cine4d(img, model_ff, progress_bar, root, stop_event)

        if stop_event.is_set():
            stopped = True
            break
        
        log_message(log_box, f'Selected slice for apex:  {slice_select[num]}/{img.shape[2] - 1}')
        log_message(log_box, f'Creating AHA segments.........')

        LVM = emp * 0
        nseg = 6
        progress_bar['value'] = 0  # Reset progress bar for inner loop
        progress_bar['maximum'] = LVM.shape[2]



        for i in range(LVM.shape[2]):
            if stop_event.is_set():
                stopped = True
                break
            if i == slice_select[num]:
                nseg = 4
            for j in range(LVM.shape[3]):
                if np.count_nonzero(emp[..., i, j] == 1) > 0 and np.count_nonzero(emp[..., i, j] == 2) > 0 and np.count_nonzero(emp[..., i, j] == 3) > 0:
                    LVM[..., i, j] = get_ahaseg(emp[..., i, j], nseg=nseg)
                else:
                    LVM[..., i, j] = (emp[..., i, j] == 2) * 1

            # Update the progress bar for the inner loop
            progress_bar['value'] = i + 1
            root.update_idletasks()  # Ensure the GUI updates

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

        dict = {'input': img, 'Seg': Seg, 'SegAHA': Seg_AHA,
                'input_crop': img[x0:x1, y0:y1], 'Seg_crop': Seg[x0:x1, y0:y1],
                'SegAHA_crop': Seg_AHA[x0:x1, y0:y1],
                'LV': LV, 'LVM': LVM, 'RV': RV,
                'voxel_size': np.array(voxel_size)}
        
        dict['model'] = basename(model_ff)
        savemat(f'./output/{name}_pred_{onnx_version}.mat', dict, do_compression=True)
        log_message(log_box, f'{num + 1}/{len(files)}: {basename(file)} finished ......')
        root.after(0, update_mat_listbox)

    if stopped:
        log_message(log_box, f'Jobs stopped.........')
    else:
        log_message(log_box, f'All job finished.........')
    progress_bar['value'] = 0
    root.update_idletasks()  # Ensure the GUI updates
    root.after(0, update_mat_listbox)


def on_mat_select(event):
    global seg, data
    widget = event.widget
    selection = widget.curselection()
    if selection:
        selected_mat = widget.get(selection[0])
        if selected_mat:
            mat_path = os.path.join(output_path, selected_mat)
            try:
                data = loadmat(mat_path)
                selected_type = display_type_combo.get()
                on_display_type_change(0)
                log_message(log_box, f"Showing {selected_mat}")
                log_message(log_box, f"{selected_type} matrix size: {seg.shape}")                
                
                if 'model' in data:
                    model_name = data['model'][0]  # from .mat file, the string stored into a cell array
                    log_message(log_box, f"Predicted using {model_name}")
            except Exception as e:
                log_message(log_box, f"An error occurred: {e}")


def on_display_type_change(event):
    global seg, data, norm_max
    
    if data is not None:

        norm_max = np.max(data['input'][data['Seg']==1])
        if norm_max == 0:
            np.max(data['input'])
        selected_type = display_type_combo.get()

        if selected_type == 'edge':
            seg = get_edge(data['input'], data['Seg'], norm_max)
        elif selected_type == 'edge_crop':
            seg = get_edge(data['input_crop'], data['Seg_crop'], norm_max)
        else:                                  
            seg = data[selected_type]
        show_montage(seg)
        update_time_slider(seg)  # Adapt the range of the time points
    try:
        pass
    except:
        log_message(log_box, f"Select a correct result file....")

def on_colormap_change(event):
    if seg is not None:
        show_montage(seg)  # Redraw the figure with the selected colormap

def show_montage(emp, time_frame=0):
    global fig, ax, canvas, im, norm_max
    plt.close('all')  # Close all previous figures

    # Resize the mosaic to 400x600
    height = 300
    width = 600

    # Initialize the figure and axes if they do not exist
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.axis('off')  # Hide the axes
        ax = fig.add_axes([0, 0, 1, 1])  # Remove all margins and padding
        ax.set_facecolor('black')
        fig.set_facecolor('black')

    # Clear previous canvas content
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    cmap = colormap_combo.get()  # Get the selected colormap

    # Create the padded mosaic
    padded_mosaic = create_padded_mosaic(emp, time_frame, height / width)
    selected_type = display_type_combo.get()
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
        display_max = norm_max

    if ('Seg' in selected_type) or (selected_type in ['RV', 'LV', 'LVM']):
        display_min = emp.min()
        display_max = emp.max()

    if im is None:
        im = ax.imshow(mosaic_resized, cmap=cmap, interpolation='nearest')
        im.set_clim(vmin=display_min, vmax=display_max)
    else:
        im.set_data(mosaic_resized)
        im.set_cmap(cmap)
        im.set_clim(vmin=display_min, vmax=display_max)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
    canvas.draw()


def update_montage(time_frame):
    global seg
    show_montage(seg, time_frame)

def update_time_slider(emp):
    if len(emp.shape) == 3: emp = emp[..., None]
    max_time_frame = emp.shape[3] - 1  # Get the maximum time frame
    time_slider.config(to=max_time_frame)  # Update the slider's range
    time_slider.set(0)  # Set initial value to 0

def update_mat_listbox():
    mat_files = list_mat_files(output_path)
    mat_listbox.delete(0, tk.END)
    for file in mat_files:
        mat_listbox.insert(tk.END, file)

def on_closing():
    plt.close('all')  # Close all matplotlib figures
    root.destroy()
    sys.exit()  # Ensure the program ends

def stop_processing():
    log_message(log_box, "Processing stopped by user.")
    stop_event.set()

# Create the main window
root = tk.Tk()
root.title("TigerHx GUI")

# Set default font size
default_font = ("Arial", 10)
root.option_add("*Font", default_font)

# Adjust the size of the main window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 1100
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Handle window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a frame for the combo box and label
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)

# Create a frame for the combo box and "Go" button
combo_frame = tk.Frame(frame)
combo_frame.pack(pady=5)

# Create a label for the combo box
label = tk.Label(combo_frame, text="Model")
label.pack(side=tk.LEFT, pady=5)

# Create a combo box to display the ONNX files
combo = ttk.Combobox(combo_frame, values=list_onnx_files(model_path), width=30)
combo.pack(side=tk.LEFT, padx=5)
combo.current(0)  # Select the first ONNX file by default

# Create a frame for the combo box and "Go" button
button_frame = tk.Frame(frame)
button_frame.pack(pady=5)

# Create a button to select a folder
select_folder_button = tk.Button(button_frame, text="GenCSV", command=select_folder)
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

log_box = tk.Text(log_frame, width=50, height=10, wrap=tk.WORD)
log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

log_scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_box.yview)
log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
log_box.config(yscrollcommand=log_scrollbar.set)

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
display_types = ['input', 'edge', 'Seg', 'SegAHA',
                 'input_crop', 'edge_crop', 'Seg_crop',
                 'SegAHA_crop', 'LV', 'LVM', 'RV']
display_type_combo = ttk.Combobox(display_type_frame, values=display_types, width=16)
display_type_combo.pack(side=tk.LEFT, padx=5)
display_type_combo.current(0)  # Set default display type to 'Seg'
display_type_combo.bind("<<ComboboxSelected>>", on_display_type_change)

# Create a label for the colormap combo box
colormap_label = tk.Label(display_type_frame, text="Colormap")
colormap_label.pack(side=tk.LEFT, padx=5)

# Create a combo box for selecting colormap
colormap_combo = ttk.Combobox(display_type_frame, values=['gray', 'viridis'], width=10)
colormap_combo.pack(side=tk.LEFT, padx=5)
colormap_combo.current(0)  # Set default colormap to 'gray'
colormap_combo.bind("<<ComboboxSelected>>", on_colormap_change)

# Create a scrollbar for the listbox
listbox_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a listbox for the .mat files
mat_listbox = tk.Listbox(listbox_frame, width=40, height=5, yscrollcommand=listbox_scrollbar.set)
mat_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
mat_listbox.bind('<<ListboxSelect>>', on_mat_select)

# Configure the scrollbar to work with the listbox
listbox_scrollbar.config(command=mat_listbox.yview)

# Create a progress bar
progress_bar = ttk.Progressbar(frame, mode='determinate')
progress_bar.pack(side=tk.TOP, fill=tk.BOTH)

# Create a frame for the canvas and slider
canvas_slider_frame = tk.Frame(root, width=650, height=450)
canvas_slider_frame.pack(padx=2, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=False)

# Create a frame for the canvas
canvas_frame = tk.Frame(canvas_slider_frame, width=600, height=300, bg='black')  # Set background to black and fix size
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

# Create the time slider
time_slider = tk.Scale(canvas_slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=lambda val: update_montage(int(val)))
time_slider.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

# Initial update of the .mat listbox
update_mat_listbox()

welcome_msg = f'''\n* Step 1: Click GenCSV to search Cine NIFTI files.
* Step 2: TigerHx will generate a CSV file in {sample_path}.
* Step 3: Edit the CSV file and assign APEX numbers.
* Step 4: Click 'RUN' to start the automatic segmentation.
* Step 5: Click the prediction files to inspect the results.'''
log_message(log_box, welcome_msg)

welcome_msg = f'''\n* 1: 點選GenCSV去搜尋 Cine NIfTI 資料集
* 2: TigerHx 將在 {sample_path} 生成一個 CSV 檔案
* 3: 編輯 CSV 檔案並分配 APEX 編號
* 4: 點擊 RUN 開始使用 TigerHx 進行自動心臟分割
* 5: 點擊預測檔案以檢查結果\n'''
log_message(log_box, welcome_msg)

# Run the application
root.mainloop()
