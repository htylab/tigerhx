# tool.py

import os
import numpy as np
from glob import glob
from os.path import basename, join, isfile
from scipy.io import savemat, loadmat
import nibabel as nib
import onnxruntime
from tkinter import simpledialog
import tkinter as tk
from tigerhx import lib_tool
import pandas as pd


nib.Nifti1Header.quaternion_threshold = -100

def list_onnx_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.onnx')]

def list_mat_files(mat_dir):
    return [basename(f) for f in glob(join(mat_dir, '*pred*.mat'))]

def log_message(log_box, message):
    log_box.insert(tk.END, message + '\n')
    log_box.see(tk.END)

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

def get_ahaseg(mask, nseg=6):
    from scipy import ndimage

    def mid_to_angles(mid, seg_num):
        anglelist = np.zeros(seg_num)
        if seg_num == 4:
            anglelist[0] = mid - 45 - 90
            anglelist[1] = mid - 45
            anglelist[2] = mid + 45
            anglelist[3] = mid + 45 + 90

        if seg_num == 6:
            anglelist[0] = mid - 120
            anglelist[1] = mid - 60
            anglelist[2] = mid
            anglelist[3] = mid + 60
            anglelist[4] = mid + 120
            anglelist[5] = mid + 180
        anglelist = (anglelist + 360) % 360

        angles = np.append(anglelist, anglelist[0])
        angles = np.rad2deg(np.unwrap(np.deg2rad(angles)))
        return angles.astype(int)

    def circular_sector(theta_range, lvb):
        cx, cy = ndimage.center_of_mass(lvb)
        max_range = np.min(np.abs(lvb.shape - np.array([cx, cy]))).astype(np.int64)
        r_range = np.arange(0, max_range, 0.1)
        theta = theta_range / 180 * np.pi
        z = r_range.reshape(-1, 1).dot(np.exp(1.0j * theta).reshape(1, -1))
        xall = -np.imag(z) + cx
        yall = np.real(z) + cy

        smask = lvb * 0
        xall = np.round(xall.flatten())
        yall = np.round(yall.flatten())
        mask = (xall >= 0) & (yall >= 0) & (xall < lvb.shape[0]) & (yall < lvb.shape[1])
        xall = xall[np.nonzero(mask)].astype(int)
        yall = yall[np.nonzero(mask)].astype(int)
        smask[xall, yall] = 1
        return smask

    lvb = (mask == 1)
    lvw = (mask == 2)
    rvb = (mask == 3)
    lx, ly = ndimage.center_of_mass(lvb)
    rx, ry = ndimage.center_of_mass(rvb)
    j = (-1) ** 0.5
    lvc = lx + j * ly
    rvc = rx + j * ry
    mid_angle = np.angle(rvc - lvc) / np.pi * 180 - 90

    angles = mid_to_angles(mid_angle, nseg)
    AHA_sector = lvw * 0
    for ii in range(angles.size - 1):
        angle_range = np.arange(angles[ii], angles[ii + 1], 0.1)
        smask = circular_sector(angle_range, lvb)
        AHA_sector[smask > 0] = (ii + 1)

    label_mask = AHA_sector * lvw
    return label_mask

def predict_cine4d(img, model_ff, progress_bar, root, stop_event):
    xyzt_mode = basename(model_ff).split('_')[2]
    session = onnxruntime.InferenceSession(model_ff)
    data = img.copy()

    def getLarea(input_mask):
        from scipy import ndimage
        try:
            labeled_mask, cc_num = ndimage.label(input_mask)
            mask = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
        except:
            mask = input_mask
        return mask

    def post(mask):
        masknew = mask * 0
        for jj in range(1, 4):
            masknew[getLarea(mask == jj)] = jj
        return masknew

    xx, yy, zz, tt = data.shape
    if xyzt_mode == 'xy':
        data = np.reshape(data, [xx, yy, zz * tt])

    mask_pred4d = data * 0
    for tti in range(data.shape[-1]):
        if stop_event.is_set():
            return 0

        image_raw = data[..., tti]
        image = image_raw[None, ...][None, ...]
        if np.max(image) == 0:
            continue
        image = image / np.max(image)
        logits = session.run(None, {"modelInput": image.astype(np.float32)})[0]
        mask_pred = post(np.argmax(logits[0, ...], axis=0))
        mask_pred4d[..., tti] = mask_pred

        if progress_bar:
            root.after(0, progress_bar.step, 100 / data.shape[-1])

    mask_pred4d = mask_pred4d.reshape((xx, yy, zz, tt))
    return mask_pred4d


def run_program_gui_interaction(selected_file, log_box, root):

    if selected_file.endswith(('.nii', '.nii.gz')):
        # Process NIfTI files
        files = [selected_file]
        slice_select = []
        aha4_start = -1
        log_message(log_box, f"Processing {selected_file}")
        root.update()  # Ensure the main window stays updated
        root.lift()    # Keep the main window behind the dialog
        img = nib.load(selected_file)
        while aha4_start < 0 or aha4_start > (img.shape[2] - 1):
            try:
                msg = f'   Tell me the first slice of apex 0~{img.shape[2] - 1}   '
                aha4_start = simpledialog.askinteger("Input", msg, minvalue=0, maxvalue=img.shape[2] - 1, parent=root)
                if aha4_start is None:
                    aha4_start = -1
            except:
                aha4_start = -1
        slice_select.append(aha4_start)

        return files, slice_select, None

    elif selected_file.endswith('.csv'):
        # Process CSV files
        common_path = None
        if isfile(selected_file):
  
            csv_data = pd.read_csv(selected_file)
            files = csv_data['Filename'].tolist()
            slice_select = csv_data['Apex'].tolist()


            common_path = os.path.commonpath(files)

        log_message(log_box, f"Found {len(files)} for processing....")
        return files, slice_select, common_path

    else:
        log_message(log_box, "Unsupported file type selected.")
        return None, None, None

def init_app(application_path):
    model_path = join(application_path, 'models')
    output_path = join(application_path, 'output')
    sample_path = join(application_path, 'csv')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    model_server = 'https://github.com/htylab/tigerhx/releases/download/modelhub/'

    default_models = ['cine4d_v0001_xyz_mms12.onnx',
                    'cine4d_v0002_xyz_mms12acdc.onnx',
                    'cine4d_v0003_xy_mms12acdc.onnx',
                    'cine4d_v0004_xy_retrain.onnx']

    for m0 in default_models:
        model_file = join(model_path, m0)
        if not isfile(model_file):
            try:
                print(f'Downloading model files....')
                model_url = model_server + m0
                print(model_url, model_file)
                lib_tool.download(model_url, model_file)
                download_ok = True
                print('Download finished...')
            except:
                download_ok = False

            if not download_ok:
                raise ValueError('Server error. Please check the model name or internet connection.')
