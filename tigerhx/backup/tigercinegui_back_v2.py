import tkinter as tk
from tkinter import ttk
import os
import sys
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from guitool import *
from os.path import join, isfile, basename
from skimage.transform import resize
import numpy as np
import glob
from scipy.io import loadmat, savemat

# Determine if the application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_path = join(application_path, 'models')
output_path = join(application_path, 'output')
sample_path = join(application_path, 'sample')
init_app(application_path)

# Global variables
log_box = None
root = None
progress_bar = None
display_type_combo = None
data = None  # Ensure data is globally accessible
fig, ax = None, None
canvas = None
im = None

def on_go():
    global progress_bar
    selected_model = combo.get()
    if selected_model:
        model_ff = os.path.join(model_path, selected_model)
        log_message(log_box, "Processing started...")
        progress_bar.pack(padx=10, pady=10, fill=tk.X)
        root.update()  # Ensure the main window stays updated

        files, slice_select = run_program_gui_interaction(model_ff, log_box, root)

        threading.Thread(target=process_files_multithreaded, args=(files, slice_select, model_ff)).start()
    else:
        log_message(log_box, "No Selection: Please select a model from the list.")

def process_files_multithreaded(files, slice_select, model_ff):
    onnx_version = basename(model_ff).split('_')[1]
    for num, file in enumerate(files):
        name = file.split('\\')[-1].split('.nii')[0]
        img_ori, affine, header = load_nii(file)
        img = img_ori.copy()
        voxel_size = header.get_zooms()

        if len(img.shape) == 3:
            img = img[..., None]
        log_message(log_box, f'{num + 1}/{len(files)}: Predicting {basename(file)} ......')
        emp = predict_cine4d(name, img, model_ff, progress_bar, root)
        
        log_message(log_box, f'Selected slice for apex:  {slice_select[num]}/{img.shape[2] - 1}')
        log_message(log_box, f'Creating AHA segments.........')

        LVM = emp * 0
        nseg = 6
        progress_bar['value'] = 0  # Reset progress bar for inner loop
        progress_bar['maximum'] = LVM.shape[2]

        for i in range(LVM.shape[2]):
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
        x0, x1 = max(0, xx.min() -10), min(mask.shape[0], xx.max() + 10)
        y0, y1 = max(0, yy.min() -10), min(mask.shape[1], yy.max() + 10)

        dict = {'input': img_ori, 'Seg': Seg, 'SegAHA': Seg_AHA,
                'input_crop': img_ori[x0:x1, y0:y1], 'Seg_crop': Seg[x0:x1, y0:y1],
                 'SegAHA_crop': Seg_AHA[x0:x1, y0:y1],
                'LV': LV , 'LVM': LVM, 'RV': RV,
                'voxel_size': np.array(voxel_size)}
        
        dict['model'] = basename(model_ff)
        savemat(f'./output/{name}_pred_{onnx_version}.mat', dict, do_compression=True)
        log_message(log_box, f'{num + 1}/{len(files)}: {basename(file)} finished ......')
        root.after(0, update_mat_listbox)


    #root.after(0, lambda: progress_bar.pack_forget())
    log_message(log_box, f'All job finished.........')
    progress_bar['value'] = 0
    root.update_idletasks()  # Ensure the GUI updates
    root.after(0, update_mat_listbox)

global seg
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
                if selected_type in data:
                    seg = data[selected_type]
                    log_message(log_box, f"Showing {selected_mat}")
                    log_message(log_box, f"{selected_type} matrix size: {seg.shape}")
                    show_montage(seg)
                    update_time_slider(seg)  # Adapt the range of the time points
                else:
                    log_message(log_box, f"'{selected_type}' not found in {selected_mat}")
                
                if 'model' in data:
                    model_name = data['model'][0] #from .mat file , the string stored into a cell array
                    log_message(log_box, f"Predicted using {model_name}")
            except Exception as e:
                log_message(log_box, f"An error occurred: {e}")

def on_display_type_change(event):
    global seg, data
    try:
        if seg is not None and data is not None:
            selected_type = display_type_combo.get()
            if selected_type in data:
                seg = data[selected_type]
                show_montage(seg)
                update_time_slider(seg)  # Adapt the range of the time points
    except:
        log_message(log_box, f"Select a result file....")
def show_montage(emp, time_frame=0):
    global fig, ax, canvas, im
    plt.close('all')  # Close all previous figures

    # Initialize the figure and axes if they do not exist
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4, 6), dpi=100)
        ax.axis('off')  # Hide the axes
        ax = fig.add_axes([0, 0, 1, 1])  # Remove all margins and padding
        ax.set_facecolor('black')
        fig.set_facecolor('black')

    # Clear previous canvas content
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Number of slices in the z-dimension
    num_slices = emp.shape[2]

    if len(emp.shape) == 3: emp = emp[..., None]

    # Determine grid size for the mosaic to match the aspect ratio 400:600
    slice_shape = emp[:, :, 0, time_frame].shape
    aspect_ratio = 600 / 400
    num_cols = int(np.ceil(np.sqrt(num_slices / aspect_ratio)))
    num_rows = int(np.ceil(num_slices / num_cols))

    # Initialize an empty array for the mosaic
    mosaic = np.zeros((num_rows * slice_shape[0], num_cols * slice_shape[1]))

    # Fill the mosaic with slices
    for i in range(num_slices):
        row = i // num_cols
        col = i % num_cols
        mosaic[row * slice_shape[0]:(row + 1) * slice_shape[0], col * slice_shape[1]:(col + 1) * slice_shape[1]] = emp[:, :, i, time_frame]

    # Pad the mosaic to maintain aspect ratio 400 (width) x 600 (height)
    mosaic_height, mosaic_width = mosaic.shape
    target_aspect_ratio = 600 / 400

    if mosaic_height / mosaic_width > target_aspect_ratio:
        new_width = int(mosaic_height / target_aspect_ratio)
        pad_width = new_width - mosaic_width
        padding = ((0, 0), (pad_width // 2, pad_width - pad_width // 2))
    else:
        new_height = int(mosaic_width * target_aspect_ratio)
        pad_height = new_height - mosaic_height
        padding = ((pad_height // 2, pad_height - pad_height // 2), (0, 0))

    padded_mosaic = np.pad(mosaic, padding, mode='constant', constant_values=0)

    # Resize the mosaic to 400x600
    mosaic_resized = resize(padded_mosaic, (600, 400), anti_aliasing=True)

    if im is None:
        im = ax.imshow(mosaic_resized)
    else:
        im.set_data(mosaic_resized)
        im.set_clim(vmin=emp.min(), vmax=emp.max())

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

# Function to list sample files and print them to the log_box
def list_and_log_sample_files(sample_dir):
    sample_files = [f for f in glob.glob(join(sample_dir, '*.nii*'))]
    log_message(log_box, "Sample Files:")
    if len(sample_files) > 0:
        log_message(log_box, f'Found samples: {len(sample_files) }')
    count = 0
    for f in sample_files:
        count += 1
        log_message(log_box, f'{count}: {f}')
        
# Create the main window
root = tk.Tk()
root.title("TigerHx GUI")

# Set default font size
default_font = ("Arial", 11)

root.option_add("*Font", default_font)

# Adjust the size of the main window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 1000
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Handle window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a frame for the combo box and label
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)

# Create a label for the combo box
label = tk.Label(frame, text="Please select a segmentation model")
label.pack(pady=5)

# Create a frame for the combo box and "Go" button
combo_frame = tk.Frame(frame)
combo_frame.pack(pady=5)

# Create a combo box to display the ONNX files
combo = ttk.Combobox(combo_frame, values=list_onnx_files(model_path), width=30)  # Adjust the width as needed
combo.pack(side=tk.LEFT, padx=5)
combo.current(0)  # Select the first ONNX file by default

# Create a "Go" button
go_button = tk.Button(combo_frame, text="Go", command=on_go)
go_button.pack(side=tk.LEFT, padx=5)

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
display_types = ['input', 'Seg', 'SegAHA',
                 'input_crop', 'Seg_crop', 'SegAHA_crop',
                 'LV', 'LVM', 'RV']
display_type_combo = ttk.Combobox(display_type_frame, values=display_types, width=16)
display_type_combo.pack(side=tk.LEFT, padx=5)
display_type_combo.current(0)  # Set default display type to 'Seg'
display_type_combo.bind("<<ComboboxSelected>>", on_display_type_change)

refresh_button = tk.Button(display_type_frame, text="Refresh Preds", command=update_mat_listbox)
refresh_button.pack(side=tk.LEFT, padx=5)

# Create a listbox for the .mat files
mat_listbox = tk.Listbox(listbox_frame, width=40, height=10)
mat_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
mat_listbox.bind('<<ListboxSelect>>', on_mat_select)

# Create a progress bar
progress_bar = ttk.Progressbar(frame, mode='determinate')
progress_bar.pack(side=tk.TOP, fill=tk.BOTH)

# Create a frame for the canvas and slider
canvas_slider_frame = tk.Frame(root)
canvas_slider_frame.pack(padx=10, pady=10, side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Create a frame for the canvas
canvas_frame = tk.Frame(canvas_slider_frame, width=400, height=600, bg='black')  # Set background to black and fix size
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

# Create spacer frame
spacer_frame = tk.Frame(canvas_slider_frame, width=20)
spacer_frame.pack(side=tk.LEFT, fill=tk.Y)

# Create the time slider
time_slider = tk.Scale(canvas_slider_frame, from_=0, to=0, orient=tk.VERTICAL,
                       command=lambda val: update_montage(int(val)))
time_slider.pack(side=tk.RIGHT, fill=tk.Y)

# Initial update of the .mat listbox
update_mat_listbox()

list_and_log_sample_files('./sample')

# Run the application
root.mainloop()

