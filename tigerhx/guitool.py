import os
import numpy as np
import glob
from os.path import basename, join, isfile
from scipy.io import savemat, loadmat
import nibabel as nib
import onnxruntime
from tkinter import simpledialog, filedialog
import tkinter as tk
from tigerhx import lib_tool
import pandas as pd
from scipy import ndimage
import numpy as np
from skimage.transform import resize
import time
nib.Nifti1Header.quaternion_threshold = -100

def list_onnx_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.onnx')]

def list_mat_files(mat_dir):
    return [basename(f) for f in glob.glob(join(mat_dir, '*pred*.mat'))]

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
            if np.sum(mask) < 50: mask = mask * 0
        except:
            mask = input_mask
        return mask.astype(bool)

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
        try:
            logits = session.run(None, {"modelInput": image.astype(np.float32)})[0]
        except:
            logits = session.run(None, {"input": image.astype(np.float32)})[0]
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
        #slice_select = []
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
        #slice_select.append(aha4_start)

        # 提問是否將結果存入原始資料夾
        save_to_orig = messagebox.askyesno("Save to Original Folder", "Do you want to save the *.mat results to the original folder?")
        save_mat = 1 if save_to_orig else 0
        
        # 提問是否輸出 .nii.gz 格式
        save_as_gz = messagebox.askyesno("Save as .nii.gz", "Do you want to output *.nii.gz format to the original folder?")
        save_nii = 1 if save_as_gz else 0
        

        option_dict = dict()
        option_dict['Filename'] = selected_file
        option_dict['Apex'] = aha4_start
        option_dict['mat_in_inputdir'] = save_mat
        option_dict['nii_in_inputdir'] = save_nii


        return files, [option_dict], None, 

    elif selected_file.endswith('.csv'):
        # Process CSV files
        common_path = None
        if isfile(selected_file):
  
            csv_data = pd.read_csv(selected_file)
            files = csv_data['Filename'].tolist()
            #options = csv_data[['Apex', 'nii_in_inputdir', 'mat_in_inputdir']].values.tolist()
            options = csv_data[['Filename', 'Apex', 'nii_in_inputdir', 'mat_in_inputdir']].to_dict(orient='records')
            #where is apex, whether you want to save nii, whether you want to save mat in input dir

            common_path = os.path.commonpath(files)

        log_message(log_box, f"Found {len(files)} for processing....")
        return files, options, common_path

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
                    'cine4d_v0004_xy_retrain.onnx',
                    'cine4d_v0005_xy_cineT1map.onnx']

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



def select_folderX(GV):

    def get_nii_files(folder_selected, keyword):
        nii_files = []
        for root, _, files in os.walk(folder_selected):
            nii_files.extend(glob.glob(os.path.join(root, '*.nii*')))

        if keyword == '':
            ffs = nii_files
        else:
            include_list, exclude_list = extract_keywords(keyword)

            ffs = []
            for ff in nii_files:
                got_file = False
                for keyword in include_list:
                    if keyword in ff:
                        got_file = True
                        break
                for keyword in exclude_list:
                    if keyword in ff:
                        got_file = False
                        break
                if got_file: ffs.append(ff)

        return ffs

    def extract_keywords(string):
        include = []
        exclude = []
        string = string.replace(' ', '')
        words = string.split(',')
        for word in words:
            word = word.strip()
            if word.startswith('+'):
                include.append(word[1:])
            elif word.startswith('-'):
                exclude.append(word[1:])
        
        return include, exclude

    folder_selected = filedialog.askdirectory()
    keyword = simpledialog.askstring("Keyword Input",
                                     "Keyword to include and then exclude. e.g., +CINE4D,-mask.",
                                     initialvalue="+ES.nii,+ED.nii,-mask")
    

    if not folder_selected: return 0

    ffs = get_nii_files(folder_selected, keyword)      

    # Check if 'files.csv' exists and modify the filename if necessary
    log_message(GV.log_box, f"Found {len(ffs)}.")
    if len(ffs) > 0:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        f_name = os.path.join(GV.csv_path, f'files_{timestamp}.csv')
        
        with open(f_name, 'w') as f:
            f.write('Filename,Apex,mat_in_inputdir,nii_in_inputdir\n')
            for ff in ffs:
                f.write(ff + ',2,1,1\n')
        log_message(GV.log_box, f"Please edit {f_name} for segmentation.")



import os
import glob
import time
from tkinter import filedialog, simpledialog, messagebox

def select_folder(GV):
    def get_nii_files(folder_selected, keyword):
        nii_files = []
        for root, _, files in os.walk(folder_selected):
            nii_files.extend(glob.glob(os.path.join(root, '*.nii*')))

        if keyword == '':
            ffs = nii_files
        else:
            include_list, exclude_list = extract_keywords(keyword)

            ffs = []
            for ff in nii_files:
                got_file = False
                for keyword in include_list:
                    if keyword in ff:
                        got_file = True
                        break
                for keyword in exclude_list:
                    if keyword in ff:
                        got_file = False
                        break
                if got_file: ffs.append(ff)

        return ffs

    def extract_keywords(string):
        include = []
        exclude = []
        string = string.replace(' ', '')
        words = string.split(',')
        for word in words:
            word = word.strip()
            if word.startswith('+'):
                include.append(word[1:])
            elif word.startswith('-'):
                exclude.append(word[1:])
        
        return include, exclude

    folder_selected = filedialog.askdirectory()
    keyword = simpledialog.askstring("Keyword Input",
                                     "Keyword to include and then exclude. e.g., +CINE4D,-mask.",
                                     initialvalue="+ES.nii,+ED.nii,-mask")
    
    if not folder_selected: 
        return 0

    ffs = get_nii_files(folder_selected, keyword)      

    # 提問是否將結果存入原始資料夾
    save_to_orig = messagebox.askyesno("Save to Original Folder", "Do you want to save the *.mat results to the original folder?")
    save_mat = 1 if save_to_orig else 0
    
    # 提問是否輸出 .nii.gz 格式
    save_as_gz = messagebox.askyesno("Save as .nii.gz", "Do you want to output *.nii.gz format to the original folder?")
    save_nii = 1 if save_as_gz else 0

    log_message(GV.log_box, f"Found {len(ffs)}.")
    if len(ffs) > 0:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        f_name = os.path.join(GV.csv_path, f'files_{timestamp}.csv')
        
        with open(f_name, 'w') as f:
            f.write('Filename,Apex,mat_in_inputdir,nii_in_inputdir\n')
            for ff in ffs:
                f.write(ff + f',2,{save_mat},{save_nii}\n')
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

