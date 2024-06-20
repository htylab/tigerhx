import glob
from os.path import join, basename, isdir
import os
import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.special import softmax

from tigerhx import lib_tool
nib.Nifti1Header.quaternion_threshold = -100


def get_mode(model_ff):
    seg_mode, version, model_str = basename(model_ff).split('_')[1:4]

    #print(seg_mode, version , model_str)

    return seg_mode, version, model_str


def post(mask, th):

    def getLarea(input_mask, th):
        from scipy import ndimage
        labeled_mask, cc_num = ndimage.label(input_mask)

        if cc_num > 0:
            mask = (labeled_mask == (np.bincount(
                labeled_mask.flat)[1:].argmax() + 1))
        else:
            mask = input_mask

        if np.sum(mask) < th: mask = np.zeros_like(mask, dtype=bool)
        return mask

    masknew = mask * 0
    for jj in range(1, int(mask.max()) + 1):
        masknew[getLarea(mask == jj, th)] = jj

    if np.sum(masknew==1) == 0: masknew = np.zeros_like(mask, dtype=int)

    return masknew


def run(model_ff, input_data, GPU, th=50):
    xyzt_mode, _, _ = get_mode(model_ff)


    data = input_data.copy()

    data4d = (len(data.shape) == 4)

    if data4d:
        xx, yy, zz, tt = data.shape

        data = np.reshape(data, [xx, yy, zz*tt])
    else:
        xx, yy, zz = data.shape

    
    mask_pred3d = data * 0
    
    for tti in range(data.shape[-1]):

        image_raw = data[..., tti]
        image = image_raw[None, ...][None, ...]
        if np.max(image) == 0:
            continue
        image = image/np.max(image)

        logits = lib_tool.predict(model_ff, image, GPU)
   
        mask_pred = post(np.argmax(logits[0, ...], axis=0), th=th)
        mask_pred3d[..., tti] = mask_pred

    if data4d:
        mask_pred_final = np.reshape(mask_pred3d, [xx, yy, zz, tt])
    else:
        mask_pred_final = mask_pred3d


    #mask_pred_final = post(mask_pred_final, th)

    return mask_pred_final

