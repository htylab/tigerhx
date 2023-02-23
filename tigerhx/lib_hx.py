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


def post(mask):

    def getLarea(input_mask):
        from scipy import ndimage
        labeled_mask, cc_num = ndimage.label(input_mask)

        if cc_num > 0:
            mask = (labeled_mask == (np.bincount(
                labeled_mask.flat)[1:].argmax() + 1))
        else:
            mask = input_mask
        return mask

    masknew = mask * 0
    for jj in range(1, int(mask.max()) + 1):
        masknew[getLarea(mask == jj)] = jj

    return masknew


def run(model_ff, input_data, GPU):
    xyzt_mode, _, _ = get_mode(model_ff)


    data = input_data.copy()
    xx, yy, zz, tt = data.shape

    if xyzt_mode == 'xyt':
        data = np.transpose(data, [0, 1, 3, 2])

    if (xyzt_mode == 'xy') or (xyzt_mode == 'xy2'):
        xx, yy, zz, tt = data.shape
        data = np.reshape(data, [xx, yy, zz*tt])

    #affine = temp.affine
    #zoom = temp.header.get_zooms()

    
    mask_pred4d = data * 0
    mask_softmax4d = np.zeros(np.insert(data.shape, 0, 4))

    
    for tti in range(data.shape[-1]):

        image_raw = data[..., tti]
        image = image_raw[None, ...][None, ...]
        if np.max(image) == 0:
            continue
        image = image/np.max(image)

        logits = lib_tool.predict(model_ff, image, GPU)
   
        #logits = session.run(None, {"modelInput": image.astype(np.float32)})[0]
        if xyzt_mode == 'xy2':
            mask_pred = post(np.argmax(logits[0, 1:5, ...], axis=0))
        else:
            mask_pred = post(np.argmax(logits[0, ...], axis=0))
        #mask_softmax = softmax(logits[0, ...], axis=0)

        #print(xyzt_mode, tti, image.max(), mask_pred.max(), image.shape)

        mask_pred4d[..., tti] = mask_pred
        #mask_softmax4d[..., tti] = mask_softmax


    if xyzt_mode == 'xyt':
        mask_pred4d = np.transpose(mask_pred4d, [0, 1, 3, 2])
        #mask_softmax4d = np.transpose(mask_softmax4d, [0, 1, 2, 4, 3])

    if (xyzt_mode == 'xy') or (xyzt_mode == 'xy2'):
        mask_pred4d = np.reshape(mask_pred4d, [xx, yy, zz, tt])
        #mask_softmax4d = np.reshape(mask_softmax4d, [4, xx, yy, zz, tt])

    mask_pred4d = post(mask_pred4d)

    return mask_pred4d

