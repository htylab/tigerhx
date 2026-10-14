"""File input/output: NIfTI, legacy .mat results and CSV case lists."""
from __future__ import annotations

import csv
import io
import re
from os.path import basename

import nibabel as nib
import numpy as np

from tigerhx.core import aha

nib.Nifti1Header.quaternion_threshold = -100


def strip_nii_ext(path):
    """'a.b.cine.nii.gz' -> 'a.b.cine' (only the NIfTI extension is removed)."""
    name = basename(path)
    for ext in ('.nii.gz', '.nii'):
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return name


def safe_name(text):
    return re.sub(r'[^\w.\-]+', '_', text).strip('._') or 'case'


def load_nifti(path):
    """(data float32, affine, header, zooms) with data always 4D and
    zooms padded to 4 values (x, y, slice spacing, frame)."""
    nimg = nib.load(path)
    data = nimg.get_fdata(dtype=np.float32)
    zooms = [float(z) for z in nimg.header.get_zooms()][:4]
    zooms += [1.0] * (4 - len(zooms))
    while data.ndim < 4:
        data = data[..., None]
    if data.ndim > 4:
        raise ValueError(f'Expected a 3D or 4D image, got shape {data.shape}')
    return data, nimg.affine, nimg.header, zooms


def save_labels(labels, affine, header, path):
    """Save a uint8 label map with the geometry of the source image."""
    hdr = header.copy() if header is not None else None
    img = nib.Nifti1Image(np.asarray(labels, dtype=np.uint8), affine, hdr)
    img.set_data_dtype(np.uint8)
    img.header.set_slope_inter(1, 0)
    nib.save(img, path)


def display_nifti_bytes(data, zooms, dtype=None):
    """Uncompressed NIfTI bytes with an axis-aligned affine for the viewer.

    Voxel (i, j, k) maps to screen row i, column j and slice k in an axial
    view, which matches how the original GUI drew arrays with imshow.
    """
    dx, dy, dz = zooms[:3]
    affine = np.array([[0, dy, 0, 0],
                       [-dx, 0, 0, 0],
                       [0, 0, dz, 0],
                       [0, 0, 0, 1]], dtype=float)
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    elif arr.dtype.kind == 'f' and np.all(np.mod(arr, 1) == 0) \
            and arr.min() >= -32768 and arr.max() <= 32767:
        arr = arr.astype(np.int16)
    img = nib.Nifti1Image(arr, affine)
    img.header.set_xyzt_units('mm', 'sec')
    return img.to_bytes()


def crop_box(seg, margin=10):
    """[x0, x1, y0, y1] around every label in any slice and frame, plus a
    margin in pixels (the original GUI's '_crop' views); the full image when
    there is no label."""
    seg = np.asarray(seg)
    mask = seg.reshape(seg.shape[0], seg.shape[1], -1).any(axis=2)
    if not mask.any():
        return [0, int(seg.shape[0]), 0, int(seg.shape[1])]
    xs, ys = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    return [max(0, int(xs[0]) - margin), min(seg.shape[0], int(xs[-1]) + 1 + margin),
            max(0, int(ys[0]) - margin), min(seg.shape[1], int(ys[-1]) + 1 + margin)]


def label_edges(labels):
    """In-plane boundary pixels of every label, keeping the label value, for a
    contour overlay (the original GUI's 'edge' view, coloured per label)."""
    lab = np.asarray(labels)
    edge = np.zeros(lab.shape, dtype=bool)
    for axis in (0, 1):
        diff = np.diff(lab, axis=axis) != 0
        lo, hi = [slice(None)] * lab.ndim, [slice(None)] * lab.ndim
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        edge[tuple(lo)] |= diff
        edge[tuple(hi)] |= diff
    return np.where(edge & (lab > 0), lab, 0).astype(np.uint8)


def legacy_result(img, seg, apex_slice, base_first, model_name, voxel_size, ed=None, es=None):
    """The result dictionary written by the original GUI, field for field,
    plus 'AHA17'.  img and seg are 4D; voxel_size as header.get_zooms().
    ed/es default to the original rule (max/min LV-blood voxel count)."""
    seg = seg.astype(np.int64)
    lvm = aha.legacy_lvm(seg, apex_slice, base_first)
    seg_aha = aha.legacy_seg_aha(seg, lvm)
    img = np.asarray(img, dtype=np.float64)

    mask = seg.max(axis=(2, 3))
    if mask.any():
        xx, yy = np.nonzero(mask)
        x0, x1 = max(0, xx.min() - 10), min(mask.shape[0], xx.max() + 10)
        y0, y1 = max(0, yy.min() - 10), min(mask.shape[1], yy.max() + 10)
    else:
        x0, x1, y0, y1 = 0, mask.shape[0], 0, mask.shape[1]

    r = {'input_img': img, 'Seg': seg, 'SegAHA': seg_aha,
         'input_crop': img[x0:x1, y0:y1], 'Seg_crop': seg[x0:x1, y0:y1],
         'SegAHA_crop': seg_aha[x0:x1, y0:y1],
         'LV': (seg == aha.LV).astype(np.int64), 'LVM': lvm,
         'RV': (seg == aha.RV).astype(np.int64),
         'voxel_size': np.array(list(voxel_size) + [1.0] * (4 - len(voxel_size))),
         'model': basename(model_name)}
    if seg.shape[3] > 1:
        lv_curve = (seg == aha.LV).sum(axis=(0, 1, 2))
        ed = int(np.argmax(lv_curve)) if ed is None else int(ed)
        es = int(np.argmin(lv_curve)) if es is None else int(es)
        r.update({'ES_frame_0base': es, 'ED_frame_0base': ed,
                  'Seg_ES': seg[..., es], 'Seg_ED': seg[..., ed],
                  'SegAHA_ES': seg_aha[..., es], 'SegAHA_ED': seg_aha[..., ed]})
        r['AHA17'] = aha.aha17(seg, apex_slice, base_first, ed).astype(np.int64)
    else:
        r['AHA17'] = aha.aha17(seg, apex_slice, base_first, 0).astype(np.int64)
    return r


def save_mat(path, rdict):
    from scipy.io import savemat
    savemat(path, rdict, do_compression=True)


CASE_COLUMNS = ['Filename', 'Apex', 'mat_in_inputdir', 'nii_in_inputdir']


def write_csv(path, header, rows):
    """UTF-8 with BOM so Excel shows non-ASCII paths; fields are quoted as
    needed so paths containing commas survive."""
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _decode(raw):
    for enc in ('utf-8-sig', 'cp950', 'gbk', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue


def read_csv(path):
    """Rows as dicts.  Accepts UTF-8 (with/without BOM) or Big5/cp950 files
    and ',' or ';' delimiters, as written by Excel in different locales."""
    with open(path, 'rb') as f:
        text = _decode(f.read())
    first = text.splitlines()[0] if text else ''
    delimiter = ';' if first.count(';') > first.count(',') else ','
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    return [{(k or '').strip(): (v or '').strip() for k, v in row.items()} for row in rows]
