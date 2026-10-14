"""Per-case processing: segmentation, analysis and output files.

Segmentation is the only slow step and needs no user input.  The apex
choice only affects the AHA analysis, which is cheap to redo, so it is
decided after the user has seen the result.
"""
from __future__ import annotations

import os
from os.path import basename, dirname, isfile, join

import nibabel as nib
import numpy as np

from tigerhx.core import aha, metrics
from tigerhx.core.inference import model_version, segment
from tigerhx.core.io import (crop_box, legacy_result, load_nifti, save_labels, save_mat,
                             strip_nii_ext)


def _seg_path(project, case):
    return join(project.case_dir(case), 'seg.nii.gz')


def load_seg(project, case):
    """The case's 4D label map (x, y, slice, frame) as uint8."""
    seg = np.asarray(nib.load(_seg_path(project, case)).dataobj).astype(np.uint8)
    while seg.ndim < 4:
        seg = seg[..., None]
    return seg


def segment_case(project, case, model_path, gpu=False, progress=None, cancel=None):
    """Run the model, save the label map and analyse it with a suggested apex."""
    data, affine, header, zooms = load_nifti(case.path)
    seg = segment(data, model_path, gpu=gpu, progress=progress, cancel=cancel)
    out_dir = project.case_dir(case)
    os.makedirs(out_dir, exist_ok=True)
    orig_ndim = len(nib.load(case.path).shape)
    save_labels(seg if orig_ndim == 4 else seg[..., 0], affine, header, _seg_path(project, case))
    with project.lock:
        case.model = basename(model_path)
        case.shape = list(data.shape)
        case.zooms = zooms
        case.base_first = None
        case.apex_slice = None
        case.apex_confirmed = False
    analyze_case(project, case, seg)
    write_outputs(project, case, data, seg, affine, header)
    return seg


def analyze_case(project, case, seg=None):
    """ED/ES frames, slice order, apex suggestion (unless already set), the
    AHA17 map and the segmentation checks.  Fast enough to rerun whenever the
    user changes the apex."""
    if seg is None:
        seg = load_seg(project, case)
    ed, es = metrics.find_ed_es(seg)
    with project.lock:
        base_first, apex = case.base_first, case.apex_slice
    if base_first is None:
        base_first = aha.detect_base_first(seg, ed)
    if apex is None:
        apex = aha.suggest_apex_slice(seg, ed, base_first)
    a17 = aha.aha17(seg, apex, base_first, ed)
    src = nib.load(case.path)
    save_labels(a17 if len(src.shape) == 4 else a17[..., 0], src.affine, src.header,
                join(project.case_dir(case), 'aha17.nii.gz'))
    levels = aha.slice_levels(seg, ed, apex, base_first)
    checks = metrics.slice_checks(seg, ed, es, base_first, zooms=case.zooms)
    with project.lock:
        case.ed, case.es = ed, es
        case.base_first, case.apex_slice = base_first, apex
        case.levels = [levels[z] for z in range(seg.shape[2])]
        case.checks = checks
        case.crop = crop_box(seg)
    return a17


def set_apex(project, case, apex_slice, base_first=None):
    """User-confirmed apex (and optionally slice order); reanalyse."""
    with project.lock:
        case.apex_slice = int(apex_slice)
        if base_first is not None:
            case.base_first = bool(base_first)
        case.apex_confirmed = True
        case.status = 'done'
    analyze_case(project, case)


def write_outputs(project, case, data=None, seg=None, affine=None, header=None):
    """Legacy .mat (workspace and/or input folder) and input-folder NIfTI,
    according to the project settings."""
    s = project.settings
    if not (s.get('save_mat') or s.get('mat_in_inputdir') or s.get('nii_in_inputdir')):
        return
    if data is None or affine is None:
        data, affine, header, _ = load_nifti(case.path)
    if seg is None:
        seg = load_seg(project, case)
    ver = model_version(case.model)
    stem = strip_nii_ext(case.path)
    outputs = {}
    if s.get('save_mat') or s.get('mat_in_inputdir'):
        voxel_size = list(nib.load(case.path).header.get_zooms())
        rdict = legacy_result(data, seg, case.apex_slice, case.base_first, case.model, voxel_size,
                              ed=case.ed, es=case.es)
        if s.get('save_mat'):
            p = join(project.case_dir(case), f'{stem}_pred_{ver}.mat')
            save_mat(p, rdict)
            outputs['mat'] = p
        if s.get('mat_in_inputdir'):
            p = join(dirname(case.path), f'{stem}_{ver}.mat')
            save_mat(p, rdict)
            outputs['mat_inputdir'] = p
    if s.get('nii_in_inputdir'):
        p = join(dirname(case.path), f'{stem}_{ver}_seg.nii.gz')
        orig_ndim = len(nib.load(case.path).shape)
        save_labels(seg if orig_ndim == 4 else seg[..., 0], affine, header, p)
        outputs['nii_inputdir'] = p
    with project.lock:
        case.outputs.update(outputs)


def has_seg(project, case):
    return isfile(_seg_path(project, case))
