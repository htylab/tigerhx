## Background

* This package provides deep-learning segmentation models.
* We also provided the stand-alone application working on Windows, Mac, and Linux.

![tigerhx](./doc/tigerhx.png)

### Install stand-alone version
    https://github.com/htylab/tigerhx/releases
### Install package
    pip install onnxruntime #for gpu version: onnxruntime-gpu
    pip install https://github.com/htylab/tigerhx/archive/release.zip
    
## Usage
    import tigerhx
    tigerhx.run('', r'C:\sample\cine4d.nii.gz', r'c:\temp') #Segmentation only
    tigerhx.run('r', r'C:\sample\cine4d.nii.gz', r'c:\temp')
    tigerhx.run('rg', r'C:\sample\cine4d.nii.gz', r'c:\temp') # with GPU

### As a command line tool:

    tigerhx -r c:\data\*.nii.gz -o c:\output


# Label names:

## cine4d
| Label No. | Structure Name            |
| --------- | ------------------------- |
| 1         | Left Ventricle blood      |
| 2         | Left Ventricle Myocardium |
| 3         | Right Ventricle blood     |
