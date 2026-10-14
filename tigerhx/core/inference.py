"""ONNX inference for cine short-axis segmentation."""
from __future__ import annotations

import threading
from os.path import basename

import numpy as np
from scipy import ndimage


class Cancelled(Exception):
    pass


def model_mode(model_path):
    """'xy' (slice-wise) or 'xyz' (volume-wise) from cine4d_vNNNN_<mode>_*.onnx."""
    parts = basename(model_path).split('_')
    return parts[2] if len(parts) > 2 and parts[2] in ('xy', 'xyz') else 'xy'


def model_version(model_path):
    parts = basename(model_path).split('_')
    return parts[1] if len(parts) > 1 else basename(model_path)


def keep_largest_component(mask, min_size=50):
    """Keep only the largest connected component of each label; a label whose
    largest component is smaller than min_size voxels is removed."""
    out = np.zeros_like(mask)
    for label in np.unique(mask):
        if label == 0:
            continue
        cc, n = ndimage.label(mask == label)
        if n == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        if sizes.max() < min_size:
            continue
        out[cc == sizes.argmax() + 1] = label
    return out


_sessions = {}
_sessions_lock = threading.Lock()


def get_session(model_path, gpu=False):
    """Cached onnxruntime session; GPU providers are used only if available."""
    import onnxruntime as ort
    key = (model_path, bool(gpu))
    with _sessions_lock:
        if key not in _sessions:
            providers = ['CPUExecutionProvider']
            if gpu:
                available = ort.get_available_providers()
                providers = [p for p in ('CUDAExecutionProvider', 'DmlExecutionProvider',
                                         'CoreMLExecutionProvider') if p in available] + providers
            so = ort.SessionOptions()
            so.log_severity_level = 3
            _sessions[key] = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        return _sessions[key]


def segment(img, model_path, gpu=False, progress=None, cancel=None, min_size=50):
    """Segment a 3D (x, y, slice) or 4D (x, y, slice, frame) cine image.

    progress(done, total) is called after every model call; cancel is any
    object with is_set() (e.g. threading.Event) and raises Cancelled.
    Returns uint8 labels with the same shape as img.
    """
    session = get_session(model_path, gpu)
    inp = session.get_inputs()[0]
    dtype = np.float64 if inp.type == 'tensor(double)' else np.float32
    data = np.nan_to_num(np.asarray(img, dtype=np.float32))
    squeeze = data.ndim == 3
    if squeeze:
        data = data[..., None]
    nz, nt = data.shape[2], data.shape[3]
    pred = np.zeros(data.shape, dtype=np.uint8)

    if model_mode(model_path) == 'xyz':
        jobs = [(slice(None), t) for t in range(nt)]
    else:
        jobs = [(z, t) for z in range(nz) for t in range(nt)]

    for k, (z, t) in enumerate(jobs):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        image = data[:, :, z, t]
        vmax = image.max()
        if vmax > 0:
            x = (image / vmax)[None, None].astype(dtype)
            logits = session.run(None, {inp.name: x})[0]
            pred[:, :, z, t] = keep_largest_component(
                np.argmax(logits[0], axis=0).astype(np.uint8), min_size)
        if progress is not None:
            progress(k + 1, len(jobs))

    return pred[..., 0] if squeeze else pred
