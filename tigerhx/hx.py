import sys
import os
from os.path import join, isdir, basename, dirname
import argparse
import glob
from scipy.io import savemat
import nibabel as nib
import numpy as np
import time

from tigerhx import lib_hx
from tigerhx import lib_tool



def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')

def write_file(input_file, output_dir, mask, postfix='pred'):

    if output_dir is None:
        output_dir = dirname(input_file)

    output_file = basename(input_file).replace('.nii.gz', '').replace('.nii', '')
    subfile = basename(input_file).replace(output_file, '') # get .nii.gz or .nii
    output_file = output_file + f'_{postfix}{subfile}'
    output_file = join(output_dir, output_file)
    print('Writing output file: ', output_file)

    input_nib = nib.load(input_file)

    print(input_nib.shape)
    print(mask.shape)
    result = nib.Nifti1Image(mask.astype(np.uint8), input_nib.affine, input_nib.header)
 

    nib.save(result, output_file)

    result = nib.Nifti1Image(input_nib.get_fdata().astype(int), input_nib.affine, input_nib.header)
 

    nib.save(result, output_file.replace('_hx', '_new'))


    return output_file


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('--model', default=None, type=str,
                        help='Specifies the modelname')

    args = parser.parse_args()
    run_args(args)


def run(argstring, input, output=None, model=None):

    from argparse import Namespace
    args = Namespace()

    args.gpu = 'g' in argstring
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
        #args.model = 'cine4d_xyz_v002_m12ac'
        args.model = 'cine4d_v0003_xy_mms12acdc.onnx'
        #args.model = 'cine4d_xyz_v001_m12.onnx'

    model_name = lib_tool.get_model(args.model)

    result_list = []

    for f in input_file_list:
        result_dict = dict()

        print('Predicting:', f)
        t = time.time()
        input_data = nib.load(f).get_fdata()
        mask_pred = lib_hx.run(
            model_name, input_data, GPU=args.gpu)
        mask_pred = lib_hx.post(mask_pred)
        output_file = write_file(
             f, output_dir, mask_pred, postfix='hx')

        result_dict['input'] = f
        result_dict['output'] = output_file

        result_list.append(result_dict)
        print('Processing time: %d seconds' % (time.time() - t))

    if len(result_list) == 1:
        return result_list[0]
    else:
        return result_list
    
def predict(data, model='cine4d_v0003_xy_mms12acdc.onnx', GPU=False):
    # for single call from python package
    model_name = lib_tool.get_model(model)
    mask_pred = lib_hx.run(model_name, data, GPU=GPU)
    mask_pred = lib_hx.post(mask_pred)

    return mask_pred

if __name__ == "__main__":
    main()
