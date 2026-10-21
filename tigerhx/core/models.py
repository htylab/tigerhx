"""Model files: location, listing and download."""
from __future__ import annotations

import os
import shutil
import sys
from os.path import basename, dirname, isfile, join

MODEL_SERVERS = ['https://github.com/htylab/tigerhx/releases/download/modelhub/',
                 'https://data.mrilab.org/onnxmodel/dev/',
                 'https://data.mrilab.org/onnxmodel/releasev1/']

GUI_MODELS = ['cine4d_v0004_xy_retrain.onnx',
              'cine4d_v0005_xy_cineT1map.onnx',
              'cine4d_v0006_xy_cine.onnx']


def app_dir():
    """Folder of the frozen executable, or of the tigerhx package."""
    if getattr(sys, 'frozen', False):
        return dirname(sys.executable)
    return dirname(dirname(os.path.abspath(__file__)))


def model_dir():
    path = os.environ.get('TIGERHX_MODEL_DIR') or join(app_dir(), 'models')
    os.makedirs(path, exist_ok=True)
    return path


def list_models(directory=None):
    directory = directory or model_dir()
    return sorted(f for f in os.listdir(directory) if f.endswith('.onnx'))


def download(url, file_name, progress=None):
    """Download to file_name.part and rename, so an interrupted download never
    leaves a truncated model behind.  progress(bytes_done, bytes_total)."""
    import ssl
    import urllib.request
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    part = file_name + '.part'
    try:
        with urllib.request.urlopen(url, context=context) as response, open(part, 'wb') as out:
            total = int(response.headers.get('Content-Length') or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        os.replace(part, file_name)
    finally:
        if os.path.exists(part):
            os.remove(part)


def ensure_model(name, progress=None, directory=None):
    """Path of a model, downloading it from the model servers if needed."""
    if isfile(name):
        return name
    if not name.endswith('.onnx'):
        name += '.onnx'
    path = join(directory or model_dir(), basename(name))
    if isfile(path):
        return path
    errors = []
    for server in MODEL_SERVERS:
        try:
            download(server + basename(name), path, progress)
            return path
        except Exception as e:  # try the next server
            errors.append(f'{server}: {e}')
    raise RuntimeError(f'Cannot download {name}. Check the model name or internet connection.\n'
                       + '\n'.join(errors))
