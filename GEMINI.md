# tigerhx Project Context

## Project Overview

**tigerhx** is a deep learning-based tool designed for **Tissue Mask Generation for the Heart**, specifically processing Cine MRI images. It automates the segmentation of cardiac structures including:
*   Left Ventricle blood
*   Left Ventricle Myocardium
*   Right Ventricle blood

The project offers both a Command Line Interface (CLI) and a Graphical User Interface (GUI) for ease of use. It utilizes **ONNX Runtime** for efficient deep learning model inference.

## Key Technologies

*   **Language:** Python 3.8+
*   **Core Libraries:**
    *   `numpy`: Numerical operations.
    *   `nibabel`: Reading and writing NIfTI medical image files.
    *   `scipy`: Scientific computing (image processing, .mat file I/O).
    *   `onnxruntime` / `onnxruntime-gpu`: Deep learning model inference.
*   **GUI:** `tkinter` (Standard Python GUI), `matplotlib` (Plotting).
*   **Building:** `setuptools` (Packaging), `PyInstaller` (Standalone executable creation).
*   **CI/CD:** GitHub Actions (Automated builds for Windows, Linux, macOS).

## Directory Structure

*   `tigerhx/`: Main source code directory.
    *   `hx.py`: Entry point for the CLI tool.
    *   `tigercinegui.py`: Entry point for the GUI application.
    *   `lib_hx.py`: Core processing logic (pre-processing, inference, post-processing).
    *   `lib_tool.py`: Utility functions (model downloading, CPU counting, helper functions).
    *   `guitool.py`: Helper functions for the GUI.
    *   `tigercinegui_qt.py`: Likely an alternative or legacy Qt-based GUI (current main GUI uses Tkinter).
*   `.github/workflows/`: CI/CD configurations.
    *   `python-app.yml`: Builds the project and creates standalone executables using PyInstaller.
*   `setup.py` & `pyproject.toml`: Project metadata and build configuration.

## Building and Running

### Installation

1.  **Dependencies:**
    ```bash
    pip install numpy nibabel scipy onnxruntime
    # For GPU support:
    # pip install onnxruntime-gpu
    ```
2.  **Install Package:**
    ```bash
    pip install .
    ```

### Usage

#### Command Line Interface (CLI)

The package exposes the `tigerhx` command (mapped to `tigerhx.hx:main`).

```bash
# Basic usage
tigerhx -r /path/to/data/*.nii.gz -o /path/to/output

# Arguments:
#   input: Path to input image(s) or folder.
#   -o, --output: Output directory (default: same as input).
#   -g, --gpu: Enable GPU acceleration.
#   --model: Specify a custom model name.
```

#### Graphical User Interface (GUI)

To launch the GUI from the source:

```bash
python -m tigerhx.tigercinegui
```

**GUI Workflow:**
1.  **GenCSV:** Scan a folder for Cine NIfTI files and generate a CSV list.
2.  **Edit CSV:** User edits the CSV to assign APEX numbers (if required).
3.  **Run:** Select the CSV or NIfTI files to start segmentation.
4.  **Inspect:** View results within the GUI (supports different views like Edge, Seg, AHA segments).

### Building Executables

The project uses `PyInstaller` to create standalone executables. The build process is defined in `.github/workflows/python-app.yml`.

**Example Command (Windows):**
```bash
pyinstaller -c -p ./tigerhx --icon=./tigerhx/exe/ico.ico --add-data "./tigerhx/exe/onnxruntime_providers_shared.dll;onnxruntime/capi" -F ./tigerhx/hx.py
```

## Development Conventions

*   **Model Management:** Models are downloaded automatically from GitHub releases if not present locally (`lib_tool.py`).
*   **Data Handling:** Supports NIfTI (`.nii`, `.nii.gz`) and MATLAB (`.mat`) formats.
*   **Versioning:** Version is defined in `setup.py`.
*   **Testing:** Currently, the CI workflow focuses on building executables rather than running a comprehensive unit test suite (though `python-app.yml` implies standard build checks).

## Label Definitions

| Label No. | Structure Name            |
| :---: | :------------------------ |
| 1         | Left Ventricle blood      |
| 2         | Left Ventricle Myocardium |
| 3         | Right Ventricle blood     |
