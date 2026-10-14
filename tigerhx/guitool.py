"""Helpers for the Tk GUIs (tigercinegui.py, tigercinegui_v2.py).

Segmentation and AHA sectors come from tigerhx.core, shared with the CLI
and the web GUI.
"""
import glob
import os
import time
from os.path import basename, dirname, isfile, join
from tkinter import filedialog, messagebox, simpledialog

import nibabel as nib
import numpy as np
from scipy import ndimage
import tkinter as tk

from tigerhx.core import aha, io, models
from tigerhx.core.inference import Cancelled, segment

nib.Nifti1Header.quaternion_threshold = -100


def list_onnx_files(directory):
    return models.list_models(directory)


def list_mat_files(mat_dir):
    return sorted(basename(f) for f in glob.glob(join(mat_dir, '*pred*.mat')))


def log_message(log_box, message):
    log_box.insert(tk.END, message + '\n')
    log_box.see(tk.END)


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def get_ahaseg(mask, nseg=6):
    return aha.sector_map(mask, nseg)


def predict_cine4d(img, model_ff, progress_bar, root, stop_event):
    """4D segmentation; returns 0 when stopped (callers check stop_event)."""
    def progress(done, total):
        if progress_bar:
            root.after(0, lambda: progress_bar.configure(maximum=100, value=100 * done / total))
    try:
        return segment(img, model_ff, progress=progress, cancel=stop_event)
    except Cancelled:
        return 0


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def run_program_gui_interaction(selected_file, log_box, root):

    if selected_file.endswith(('.nii', '.nii.gz')):
        log_message(log_box, f"Processing {selected_file}")
        root.update()
        root.lift()
        nz = nib.load(selected_file).shape[2]
        aha4_start = simpledialog.askinteger(
            "Input", f'   Tell me the first slice of apex 0~{nz - 1}   ',
            minvalue=0, maxvalue=nz - 1, parent=root)
        if aha4_start is None:
            log_message(log_box, "Cancelled.")
            return None, None, None

        save_mat = int(messagebox.askyesno(
            "Save to Original Folder", "Do you want to save the *.mat results to the original folder?"))
        save_nii = int(messagebox.askyesno(
            "Save as .nii.gz", "Do you want to output *.nii.gz format to the original folder?"))

        option_dict = {'Filename': selected_file, 'Apex': aha4_start,
                       'mat_in_inputdir': save_mat, 'nii_in_inputdir': save_nii}
        return [selected_file], [option_dict], None

    elif selected_file.endswith('.csv') and isfile(selected_file):
        rows = [r for r in io.read_csv(selected_file) if r.get('Filename')]
        missing = [r['Filename'] for r in rows if not isfile(r['Filename'])]
        for f in missing:
            log_message(log_box, f"File not found, skipped: {f}")
        rows = [r for r in rows if isfile(r['Filename'])]
        files = [r['Filename'] for r in rows]
        options = [{'Filename': r['Filename'], 'Apex': _int(r.get('Apex'), -1),
                    'nii_in_inputdir': _int(r.get('nii_in_inputdir')),
                    'mat_in_inputdir': _int(r.get('mat_in_inputdir'))} for r in rows]
        # Folders, not files: a single-row CSV must not yield an empty name.
        common_path = os.path.commonpath([dirname(f) for f in files]) if files else None
        log_message(log_box, f"Found {len(files)} for processing....")
        return files, options, common_path

    else:
        log_message(log_box, "Unsupported file type selected.")
        return None, None, None


def init_app(application_path):
    model_path = join(application_path, 'models')
    for d in ('models', 'output', 'csv'):
        os.makedirs(join(application_path, d), exist_ok=True)

    for m0 in models.GUI_MODELS:
        if isfile(join(model_path, m0)):
            continue
        try:
            print(f'Downloading {m0} ....')
            models.ensure_model(m0, directory=model_path)
            print('Download finished...')
        except Exception as e:  # the GUI can still use the models it has
            print(f'Warning: {e}')


def get_edge(input_image, mask, norm_max):
    def edge2d(slice_mask):
        #labels = np.unique(slice_mask)
        edges_combined = np.zeros_like(slice_mask, dtype=bool)
        for label in [1, 2, 3]:
            binary_mask = (slice_mask == label).astype(int)
            sobel_x = ndimage.sobel(binary_mask, axis=0)
            sobel_y = ndimage.sobel(binary_mask, axis=1)
            edges = np.hypot(sobel_x, sobel_y)            

            edges = (edges > 2).astype(np.uint8)
            edges_combined = np.logical_or(edges_combined, edges)
        return edges_combined

    image = np.clip(input_image, 0, norm_max)
    image = (image/norm_max * 254).astype(np.uint8)

    output_image = image.copy()    
    z_dim, t_dim = image.shape[2], image.shape[3]    
    for z in range(z_dim):
        for t in range(t_dim):
            #sobel_x = ndimage.sobel(mask[:, :, z, t], axis=0)
            #sobel_y = ndimage.sobel(mask[:, :, z, t], axis=1)
            #edges = np.hypot(sobel_x, sobel_y)            

            #edges = (edges > 2).astype(np.uint8)


            edges = edge2d(mask[:, :, z, t])

            output_image[:, :, z, t][edges > 0] = 255


    return output_image


def select_folder(GV):
    folder_selected = filedialog.askdirectory()
    if not folder_selected:
        return 0
    keyword = simpledialog.askstring(
        "Keyword Input",
        "Keyword to include and then exclude. e.g., +CINE4D,-mask.",
        initialvalue="+ES.nii,+ED.nii,-mask")
    if keyword is None:
        return 0

    include = [w[1:] for w in keyword.replace(' ', '').split(',') if w.startswith('+')]
    exclude = [w[1:] for w in keyword.replace(' ', '').split(',') if w.startswith('-')]
    ffs = []
    for root, _, files in os.walk(folder_selected):
        for f in files:
            if not f.lower().endswith(('.nii', '.nii.gz')):
                continue
            ff = join(root, f)
            rel = os.path.relpath(ff, folder_selected)
            if include and not any(k in rel for k in include):
                continue
            if any(k in rel for k in exclude):
                continue
            ffs.append(ff)
    ffs.sort()

    save_mat = int(messagebox.askyesno(
        "Save to Original Folder", "Do you want to save the *.mat results to the original folder?"))
    save_nii = int(messagebox.askyesno(
        "Save as .nii.gz", "Do you want to output *.nii.gz format to the original folder?"))

    log_message(GV.log_box, f"Found {len(ffs)}.")
    if len(ffs) > 0:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        f_name = os.path.join(GV.csv_path, f'files_{timestamp}.csv')
        io.write_csv(f_name, io.CASE_COLUMNS, [[ff, 2, save_mat, save_nii] for ff in ffs])
        log_message(GV.log_box, f"Please edit {f_name} for segmentation.")


def create_padded_mosaic(emp, time_frame=0, aspect_ratio=0.66):
    # Number of slices in the z-dimension
    num_slices = emp.shape[2]

    if len(emp.shape) == 3: emp = emp[..., None]

    # Determine grid size for the mosaic to match the aspect ratio 400:600
    slice_shape = emp[:, :, 0, time_frame].shape
    #aspect_ratio = 600 / 400
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
    #target_aspect_ratio = 600 / 400
    target_aspect_ratio = aspect_ratio

    if mosaic_height / mosaic_width > target_aspect_ratio:
        new_width = int(mosaic_height / target_aspect_ratio)
        pad_width = new_width - mosaic_width
        padding = ((0, 0), (pad_width // 2, pad_width - pad_width // 2))
    else:
        new_height = int(mosaic_width * target_aspect_ratio)
        pad_height = new_height - mosaic_height
        padding = ((pad_height // 2, pad_height - pad_height // 2), (0, 0))

    padded_mosaic = np.pad(mosaic, padding, mode='constant', constant_values=0)
    
    return padded_mosaic


def get_ESED(LV):
    curve = np.sum(LV, axis=(0, 1, 2))
    ES_t = np.argmin(curve)
    ED_t = np.argmax(curve)
    return ES_t, ED_t

