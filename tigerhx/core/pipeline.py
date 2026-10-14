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
from tigerhx.core.io import (legacy_result, load_nifti, save_labels, save_mat,
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
    """ED/ES, slice order, apex suggestion (unless already set), AHA17 map and
    metrics.  Fast enough to rerun whenever the user changes a choice."""
    if seg is None:
        seg = load_seg(project, case)
    nt = seg.shape[3]
    ed, es = metrics.find_ed_es(seg, project.settings.get('ed_rule', 'max_volume'))
    with project.lock:
        if case.ed_manual is not None and 0 <= case.ed_manual < nt:
            ed = case.ed_manual
        if case.es_manual is not None and 0 <= case.es_manual < nt:
            es = case.es_manual
        base_first = case.base_first
        apex = case.apex_slice
        excluded = {ph: {ch: list(v) for ch, v in d.items()} for ph, d in case.excluded.items()}
    if base_first is None:
        base_first = aha.detect_base_first(seg, ed)
    if apex is None:
        apex = aha.suggest_apex_slice(seg, ed, base_first)
    a17 = aha.aha17(seg, apex, base_first, ed)
    src = nib.load(case.path)
    save_labels(a17 if len(src.shape) == 4 else a17[..., 0], src.affine, src.header,
                join(project.case_dir(case), 'aha17.nii.gz'))
    zooms = case.zooms or [1.0, 1.0, 1.0]
    with project.lock:
        case.ed, case.es = ed, es
        case.base_first, case.apex_slice = base_first, apex
        levels = aha.slice_levels(seg, ed, apex, base_first)
        case.levels = [levels[z] for z in range(seg.shape[2])]
        case.metrics = metrics.compute_metrics(seg, zooms, ed, es, excluded)
        case.checks = metrics.slice_checks(seg, ed, es, base_first, excluded)
        case.curves = {k: [round(v, 2) for v in vals]
                       for k, vals in metrics.volume_curves(seg, zooms).items()}
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


def set_phases(project, case, ed=None, es=None, reset=False):
    """User-chosen ED and/or ES frame (reset=True returns to automatic)."""
    nt = case.shape[3] if case.shape else 1
    for value in (ed, es):
        if value is not None and not 0 <= int(value) < nt:
            raise ValueError(f'frame must be within 0..{nt - 1}')
    with project.lock:
        if reset:
            case.ed_manual = case.es_manual = None
        if ed is not None:
            case.ed_manual = int(ed)
        if es is not None:
            case.es_manual = int(es)
    analyze_case(project, case)


def set_excluded(project, case, phase, chamber, slice_index, exclude=True):
    """Leave one slice out of (or back into) the volume of one chamber
    (LV/RV) at one phase (ED/ES), e.g. a basal slice that is atrium at ES."""
    if phase not in case.excluded or chamber not in case.excluded[phase]:
        raise ValueError('phase must be ED or ES and chamber LV or RV')
    nz = case.shape[2] if case.shape else 0
    if not 0 <= int(slice_index) < nz:
        raise ValueError(f'slice must be within 0..{nz - 1}')
    with project.lock:
        current = set(case.excluded[phase][chamber])
        if exclude:
            current.add(int(slice_index))
        else:
            current.discard(int(slice_index))
        case.excluded[phase][chamber] = sorted(current)
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
