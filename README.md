## Background
### TigerHx: Tissue Mask Generation for Heart
* This repo provides deep learning methods with pretrained models for cardiac cine MRI segmentation.
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

    tigerhx c:\data\*.nii.gz -o c:\output
    tigerhx c:\data -o c:\output -g --model cine4d_v0006_xy_cine.onnx

## GUI

    pip install "tigerhx[gui] @ https://github.com/htylab/tigerhx/archive/main.zip"
    tigerhx-gui                      # or: python -m tigerhx.webgui.app

The GUI runs a local server (127.0.0.1 only) and shows the page in a native
window when `pywebview` is installed, otherwise in the default browser
(`--browser` forces the browser). Results and the case list are kept in a
workspace folder, `~/TigerHx` by default (`-w <folder>` to change).

Workflow:

1. **＋ Files / ＋ Folder** – add cine NIfTI files; folders are searched
   recursively with optional include / exclude keywords. No CSV is needed.
2. **▶ Segment** – runs the selected model on all pending (or the checked)
   cases. Models are downloaded on first use.
3. **Review** – the viewer (NiiVue) shows the image with the segmentation or
   AHA 17-segment overlay, single slice or all slices, frame slider, ED/ES
   jump and the LV/RV volume–time curve.
   The first apical slice and the slice order (base→apex or apex→base) are
   suggested automatically; check them on the images and use
   *Use current slice as apex* or *Confirm suggestion*. The AHA labels and
   metrics are recomputed instantly.
4. **Check ED/ES and the slices** (results panel) – replace the automatic ED
   or ES frame with the frame on screen, and at ED or ES leave individual
   slices out of the LV or RV volume, e.g. a basal slice that is already
   atrium at ES or a slice below the apex the model labelled by mistake.
   The panel lists slices worth checking: a cavity larger at ES than at ED,
   a cavity missing between two slices that have it, a basal-most LV at ES
   that is poorly enclosed by myocardium (probably atrium), and a reminder
   for the basal-most slice at ES. *Show* jumps to the frame and slice;
   cases with open warnings are marked ⚠ in the case list. Slices whose
   cavity at ES is more than twice its ED area plus 200 mm² (typically a
   structure below the apex) are left out automatically and marked *auto*;
   × counts them again. Metrics update immediately.
5. **Export metrics CSV** – LV/RV EDV, ESV, SV, EF and LV mass for all cases,
   with the ED/ES frames, how they were chosen and the excluded slices.

Files written per case in `<workspace>/cases/<name>/`: `seg.nii.gz`
(labels 1–3), `aha17.nii.gz` (AHA segments 1–17 on the myocardium) and,
unless switched off under *Settings*, the `.mat` file of the previous GUI
(`*_pred_<version>.mat`, same fields plus `AHA17`).

Notes on the analysis:

* ES is the frame with the smallest LV blood volume. ED is the frame with the
  largest LV volume (default, as in the previous GUI) or the first frame
  (*Settings → Automatic ED*; the R-wave triggered frame of an ECG-gated cine).
  Either can be overridden per case. LV and RV values are both taken at these
  ED/ES frames. LV mass = myocardium volume at ED × 1.05 g/mL. Volumes use the
  voxel size from the NIfTI header (slice spacing included); EF does not
  depend on it. The volume–time curves always include every slice.
* Post-processing was chosen on 15 ACDC cases (v0006, labelled ED/ES frames):
  the automatic ES-outlier exclusion lowered the LV EF mean absolute error
  from 4.1 to 3.1 points (worst case 21.5 → 8.4) and fired only on slices the
  experts had left empty; acting on the open-ring warnings lowered it further
  to 2.2. Hole filling and optical-flow temporal voting were tested and did
  not help, so they are not used.
* Trigger times are not needed for EF. Cardiac output and time-based indices
  (filling rates) would need the heart rate or trigger times, which NIfTI
  files usually do not carry.
* AHA sectors use the direction from the LV to the RV blood-pool centroid as
  reference (septum between sectors 2 and 3), as in the previous GUI; the
  segment names assume the standard short-axis orientation. Slices from the
  base to the apical start are split evenly into basal and mid; slices with
  myocardium but no cavity beyond it are segment 17.

The previous Tk GUI (`tigercinegui.py`) is still available.

## Validation

`tools/eval_acdc.py` downloads a preprocessed copy of ACDC from Hugging Face
(`mathpluscode/ACDC`, CC BY-NC-SA 4.0, resampled to 1 × 1 × 10 mm and cropped
around the LV) and compares models with the expert ED/ES labels:

    python tools/eval_acdc.py --per-group 2 --models cine4d_v0006_xy_cine.onnx
    python tools/eval_acdc.py --per-group 4 --ed-rule first_frame --out acdc.csv

It reports Dice per structure, LV/RV EF error, EDV and mass error and whether
the detected ED/ES frames match the labelled ones. Like the GUI it applies the
automatic ES-outlier exclusion (`--no-auto-exclude` to count every slice). Models trained on ACDC
(e.g. v0003 *mms12acdc*) score optimistically on it.

## Development

    pip install -e . onnxruntime onnx pytest
    pytest tests

`tigerhx/core` holds the UI-independent code shared by the CLI and both GUIs
(inference, AHA, metrics, file I/O, workspace). The tests use a synthetic
phantom and a tiny ONNX model, so they need no network or real data.


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

