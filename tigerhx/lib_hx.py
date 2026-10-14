import glob
from os.path import join, basename, isdir
import os
import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.special import softmax

from tigerhx import lib_tool
from tigerhx.core.inference import segment
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


def run(model_ff, input_data, GPU, th=50):
    """Segment a 3D or 4D cine image; th is the minimum component size."""
    return segment(input_data, model_ff, gpu=GPU, min_size=th)
