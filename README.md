## Background
### TigerHx: Tissue Mask Generation for Heart
* This repo provides deep learning methods with pretrained models for brain extraction.
* We also provided the stand-alone application working on Windows, Mac, and Linux.
* The software has been exclusively designed for research purposes and is not intended for any commercial application.
* The software should not be used in clinical applications.

![tigerhx](./doc/tigerhx.png)

### Install stand-alone version
    https://github.com/htylab/tigerhx/releases
### Install package
    pip install onnxruntime #for gpu version: onnxruntime-gpu
    pip install https://github.com/htylab/tigerhx/archive/main.zip
    
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

## Disclaimer

The software has been exclusively designed for research purposes and has not undergone review or approval by the Food and Drug Administration or any other agency. By using this software, you acknowledge and agree that it is not recommended nor advised for clinical applications.  You also agree to use, reproduce, create derivative works of, display, and distribute the software in compliance with all applicable governmental laws, regulations, and orders, including but not limited to those related to export and import control.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software. Use of the software is at the recipient's own risk.

