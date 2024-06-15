import tkinter as tk
from tkinter import ttk, simpledialog
import os
import sys
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tool import *
from os.path import join
from skimage.transform import resize

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))


print(application_path)

model_path = join(application_path, 'models')
output_path = join(application_path, 'output')

os.makedirs(output_path, exist_ok=True)

# Global variables
log_box = None
root = None
progress_bar = None
progress_window = None

def on_go():
    global progress_bar, progress_window
    selected_model = combo.get()
    if selected_model:
        model_ff = os.path.join(model_path, selected_model)
        progress_window = tk.Toplevel(root)
        progress_window.title("Processing...")
        progress_label = tk.Label(progress_window, text="Processing...")
        progress_label.pack(padx=10, pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='determinate')
        progress_bar.pack(padx=10, pady=10, fill=tk.X)
        progress_window.lift(root)  # Bring progress window to the foreground
        progress_window.grab_set()  # Make progress window modal
        root.update()  # Ensure the main window stays updated

        files, slice_select = run_program_gui_interaction(model_ff, log_box, root)

        threading.Thread(target=process_files_multithreaded, args=(files, slice_select, model_ff)).start()
    else:
        log_message(log_box, "No Selection: Please select a model from the list.")

def process_files_multithreaded(files, slice_select, model_path):
    global progress_window
    for num, file in enumerate(files):
        name = file.split('\\')[-1].split('.nii')[0]
        img_ori, affine, header = load_nii(file)
        img = img_ori.copy()
        voxel_size = header.get_zooms()

        if len(img.shape) == 3:
            img = img[..., None]

        emp = predict_cine4d(name, img, model_path, progress_bar, root)
        log_message(log_box, f'{name} slice select {slice_select[num]}/{img.shape[2] - 1}')

        LVM = emp * 0
        nseg = 6
        for i in range(LVM.shape[2]):
            if i == slice_select[num]:
                nseg = 4
            for j in range(LVM.shape[3]):
                if np.count_nonzero(emp[..., i, j] == 1) > 0 and np.count_nonzero(emp[..., i, j] == 2) > 0 and np.count_nonzero(emp[..., i, j] == 3) > 0:
                    LVM[..., i, j] = get_ahaseg(emp[..., i, j], nseg=nseg)
                else:
                    LVM[..., i, j] = (emp[..., i, j] == 2) * 1

        dict = {"input": img_ori, 'LV': (emp == 1) * 1, 'LVM': LVM, 'RV': (emp == 3) * 1, 'Seg': emp}
        savemat(f'./output/{name}_pred.mat', dict)

    root.after(0, progress_window.destroy)
    root.after(0, update_mat_listbox)

def on_mat_select(event):
    widget = event.widget
    selection = widget.curselection()
    if selection:
        selected_mat = widget.get(selection[0])
        if selected_mat:
            mat_path = os.path.join(output_path, selected_mat)
            try:
                data = loadmat(mat_path)
                if 'Seg' in data:
                    seg = data['Seg']
                    log_message(log_box, f"Showing {selected_mat}")
                    log_message(log_box, f"Seg matrix size: {seg.shape}")
                    show_montage(seg)
                    update_time_slider(seg)  # Adapt the range of the time points
                else:
                    log_message(log_box, f"'Seg' not found in {selected_mat}")
            except Exception as e:
                log_message(log_box, f"An error occurred: {e}")

def show_montage(emp, time_frame=0):
    plt.close('all')  # Close all previous figures
    # Clear previous canvas content
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Number of slices in the z-dimension
    num_slices = emp.shape[2]

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

    fig = plt.figure(figsize=(4, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # Remove all margins and padding
    ax.imshow(mosaic_resized)
    ax.axis('off')
    ax.set_facecolor('black')
    fig.set_facecolor('black')

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
    canvas.draw()

def update_montage(time_frame):
    selection = mat_listbox.curselection()
    if selection:
        selected_mat = mat_listbox.get(selection[0])
        if selected_mat:
            mat_path = os.path.join(output_path, selected_mat)
            data = loadmat(mat_path)
            if 'Seg' in data:
                seg = data['Seg']
                show_montage(seg, time_frame)

def update_time_slider(emp):
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

# Create the main window
root = tk.Tk()
root.title("ONNX Model Selector")

# Set default font size
default_font = ("Arial", 12)

# Apply default font to all widgets
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

# Create a combo box to display the ONNX files
combo = ttk.Combobox(frame, values=list_onnx_files(model_path))
combo.pack(pady=5)
combo.current(0)  # Select the first ONNX file by default

# Create a "Go" button
go_button = tk.Button(frame, text="Go", command=on_go)
go_button.pack(pady=5)

# Create a log box to display messages
log_frame = tk.Frame(frame)
log_frame.pack(padx=10, pady=10)

log_box = tk.Text(log_frame, width=50, height=10, wrap=tk.WORD)
log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

log_scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_box.yview)
log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
log_box.config(yscrollcommand=log_scrollbar.set)

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

# Create a listbox for the .mat files
mat_listbox_frame = tk.Frame(frame)
mat_listbox_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

mat_listbox_label = tk.Label(mat_listbox_frame, text="Prediction Files")
mat_listbox_label.pack()

mat_listbox = tk.Listbox(mat_listbox_frame, width=40, height=10)
mat_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
mat_listbox.bind('<<ListboxSelect>>', on_mat_select)

mat_scrollbar = tk.Scrollbar(mat_listbox_frame, orient=tk.VERTICAL)
mat_scrollbar.config(command=mat_listbox.yview)
mat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
mat_listbox.config(yscrollcommand=mat_scrollbar.set)

# Initial update of the .mat listbox
update_mat_listbox()

# Run the application
root.mainloop()
