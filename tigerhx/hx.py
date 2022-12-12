import sys
import os
from os.path import join, isdir, basename
import argparse
import glob
from scipy.io import savemat
import nibabel as nib
import numpy as np
import time

from tigerhx import lib_hx
from tigerhx import lib_tool


def get_report(input_file, output_file):
    
    temp = nib.load(output_file)
    mask4d = temp.get_fdata()
    voxel_size = temp.header.get_zooms()
    
    LV_vol = np.sum(mask4d==1, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    LVM_vol = np.sum(mask4d==2, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    RV_vol = np.sum(mask4d==3, axis=(0, 1, 2))* np.prod(voxel_size[0:3]) / 1000.
    
    dict1 = {"input":np.asanyarray(nib.load(input_file).dataobj),
             'LV': (mask4d==1)*1,
             'LVM':(mask4d==2)*1,
             'RV': (mask4d==3)*1,
             'LV_vol': LV_vol,
             'LVM_vol': LVM_vol,
             'RV_vol': RV_vol}

    savemat(output_file.replace('.nii.gz', '.mat'), dict1, do_compression=True)


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')

def write_file(input_file, output_dir, mask, postfix='pred'):

    if not isdir(output_dir):
        print('Output dir does not exist.')
        return 0

    output_file = basename(input_file).replace('.nii.gz', '').replace('.nii', '') 
    output_file = output_file + f'_{postfix}.nii.gz'
    output_file = join(output_dir, output_file)
    print('Writing output file: ', output_file)

    input_nib = nib.load(input_file)
    affine = input_nib.affine
    zoom = input_nib.header.get_zooms()   
    result = nib.Nifti1Image(mask.astype(np.uint8), affine)
    result.header.set_zooms(zoom)

    nib.save(result, output_file)

    return output_file


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-r', '--report', action='store_true', help='Produce additional reports')
    parser.add_argument('--model', default=None, type=str,
                        help='Specifies the modelname')

    args = parser.parse_args()
    run_args(args)


def run(argstring, input, output=None, model=None):

    from argparse import Namespace
    args = Namespace()

    args.gpu = 'g' in argstring
    args.report = 'r' in argstring
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    run_args(args)

def run_args(args):

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output

    print('Total nii files:', len(input_file_list))

    if args.model is None:
        args.model = 'cine4d_xyz_v002_m12ac'

    model_name = lib_tool.get_model(args.model)

    for f in input_file_list:

        print('Predicting:', f)
        t = time.time()
        input_data = nib.load(f).get_fdata()
        mask_pred = lib_hx.run(
            model_name, input_data, GPU=args.gpu)
        mask_pred = lib_hx.post(mask_pred)
        output_file = write_file(
             f, output_dir, mask_pred, postfix='hx')

        if args.report:
            get_report(f, output_file)

        print('Processing time: %d seconds' % (time.time() - t))    

if __name__ == "__main__":
    main()
