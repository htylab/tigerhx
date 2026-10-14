"""Ventricular volumes and function from a 4D label map (x, y, slice, frame)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from tigerhx.core.aha import LV, MYO, RV

MYOCARDIAL_DENSITY = 1.05  # g/mL

# How the ED frame is chosen when the user has not set it.
ED_RULES = ('max_volume', 'first_frame')

CHAMBER_LABELS = {'LV': LV, 'RV': RV}


def voxel_ml(zooms):
    return float(np.prod(np.asarray(zooms[:3], dtype=float))) / 1000.0


def volume_curves(seg, zooms):
    """Volume (mL) of LV blood, myocardium and RV blood for every frame
    (all slices, no exclusions)."""
    v = voxel_ml(zooms)
    return {name: ((seg == label).sum(axis=(0, 1, 2)) * v).tolist()
            for name, label in (('LV', LV), ('MYO', MYO), ('RV', RV))}


def find_ed_es(seg, rule='max_volume'):
    """ED/ES frames.  ES is the frame with the smallest LV-blood volume; ED is
    the largest ('max_volume', the original GUI rule) or frame 0
    ('first_frame', the R-wave triggered frame of an ECG-gated cine)."""
    if rule not in ED_RULES:
        raise ValueError(f'unknown ED rule {rule!r}')
    curve = (seg == LV).sum(axis=(0, 1, 2))
    ed = 0 if rule == 'first_frame' else int(np.argmax(curve))
    return ed, int(np.argmin(curve))


def included_slices(nz, excluded=()):
    """Boolean mask of the slices that are counted."""
    keep = np.ones(nz, dtype=bool)
    for z in excluded or ():
        if 0 <= z < nz:
            keep[z] = False
    return keep


def _ef(edv, esv):
    return (edv - esv) / edv * 100.0 if edv > 0 else None


def compute_metrics(seg, zooms, ed, es, excluded=None):
    """Clinical summary at the ED and ES frames, for LV and RV alike.

    excluded: {'ED': {'LV': [slices], 'RV': [...]}, 'ES': {...}} lists slices
    not counted for that phase and chamber (e.g. a basal slice that is
    already atrium at ES).  LV mass uses the LV exclusions at ED.
    EF/SV/ESV are None for single-frame data.
    """
    v = voxel_ml(zooms)
    nz = seg.shape[2]
    multi = seg.shape[3] > 1
    excluded = excluded or {}

    def vol(label, frame, phase, chamber):
        keep = included_slices(nz, (excluded.get(phase) or {}).get(chamber))
        return float((seg[:, :, keep, frame] == label).sum()) * v

    lv_ed, lv_es = vol(LV, ed, 'ED', 'LV'), vol(LV, es, 'ES', 'LV')
    rv_ed, rv_es = vol(RV, ed, 'ED', 'RV'), vol(RV, es, 'ES', 'RV')
    m = {
        'LV_EDV_mL': lv_ed,
        'LV_ESV_mL': lv_es if multi else None,
        'LV_SV_mL': lv_ed - lv_es if multi else None,
        'LV_EF_pct': _ef(lv_ed, lv_es) if multi else None,
        'LV_mass_g': vol(MYO, ed, 'ED', 'LV') * MYOCARDIAL_DENSITY,
        'RV_EDV_mL': rv_ed,
        'RV_ESV_mL': rv_es if multi else None,
        'RV_SV_mL': rv_ed - rv_es if multi else None,
        'RV_EF_pct': _ef(rv_ed, rv_es) if multi else None,
    }
    return {k: (None if val is None else round(float(val), 2)) for k, val in m.items()}


def _pixel_mm2(zooms):
    return float(zooms[0]) * float(zooms[1]) if zooms else 1.0


def _areas(seg, label, frame, px):
    """Area (mm^2) of a label on every slice of one frame."""
    return (seg[..., frame] == label).sum(axis=(0, 1)) * px


def lv_ring_coverage(slice_mask):
    """Fraction of the LV-blood border that touches myocardium (None for a
    tiny LV).  Close to 1 inside the ventricle; an open ring at the base
    suggests the slice is already atrium."""
    lv = slice_mask == LV
    if lv.sum() < 20:
        return None
    edge = lv & ~ndimage.binary_erosion(lv)
    return float((edge & ndimage.binary_dilation(slice_mask == MYO)).sum() / edge.sum())


def auto_exclusions(seg, ed, es, zooms=None, factor=2.0, margin_mm2=200.0):
    """Slices left out automatically: a cavity at ES more than `factor` times
    its ED area plus `margin_mm2` is not that ventricle (typically a structure
    below the apex).  Validated on ACDC: it fired only on such slices."""
    out = {'ED': {'LV': [], 'RV': []}, 'ES': {'LV': [], 'RV': []}}
    if seg.shape[3] < 2 or ed == es:
        return out
    px = _pixel_mm2(zooms)
    for chamber, label in CHAMBER_LABELS.items():
        a_ed, a_es = _areas(seg, label, ed, px), _areas(seg, label, es, px)
        out['ES'][chamber] = [int(z) for z in np.nonzero(a_es > factor * a_ed + margin_mm2)[0]]
    return out


def slice_checks(seg, ed, es, base_first=True, excluded=None, zooms=None):
    """Slices whose segmentation is worth a look.

    'es_gt_ed':  a cavity is clearly larger at ES than at ED on that slice
                 (often a structure outside the ventricle near the apex).
    'gap':       a cavity is (almost) missing on a slice although both
                 neighbours have it.
    'open_ring': the basal-most LV slice at ES is poorly enclosed by
                 myocardium, so the LV label probably covers atrium.
    Neighbours listed in excluded do not count as evidence of a gap.
    """
    excluded = excluded or {}
    checks = []
    if seg.shape[3] < 2 or ed == es:
        return checks
    nz = seg.shape[2]
    px = _pixel_mm2(zooms)
    min_area = 100.0  # mm^2
    for chamber, label in CHAMBER_LABELS.items():
        a = {'ED': _areas(seg, label, ed, px), 'ES': _areas(seg, label, es, px)}
        for z in range(nz):
            if a['ES'][z] > 1.1 * a['ED'][z] + 50:
                checks.append({'kind': 'es_gt_ed', 'phase': 'ES', 'chamber': chamber, 'slice': z,
                               'ed': int(round(a['ED'][z])), 'es': int(round(a['ES'][z]))})
        for ph in ('ED', 'ES'):
            skip = set((excluded.get(ph) or {}).get(chamber) or ())
            for z in range(1, nz - 1):
                if skip & {z - 1, z + 1}:
                    continue
                lo = min(a[ph][z - 1], a[ph][z + 1])
                if lo > min_area and a[ph][z] < 0.2 * lo:
                    checks.append({'kind': 'gap', 'phase': ph, 'chamber': chamber, 'slice': z,
                                   'area': int(round(a[ph][z])), 'neighbours': int(round(lo))})
        if chamber != 'LV':
            continue
        order = range(nz) if base_first else range(nz - 1, -1, -1)
        basal = next((z for z in order if a['ES'][z] > min_area), None)
        cov = None if basal is None else lv_ring_coverage(seg[:, :, basal, es])
        if cov is not None and cov < 0.75:
            checks.append({'kind': 'open_ring', 'phase': 'ES', 'chamber': chamber, 'slice': basal,
                           'coverage': round(cov, 2)})
    return checks


METRIC_KEYS = ['LV_EDV_mL', 'LV_ESV_mL', 'LV_SV_mL', 'LV_EF_pct', 'LV_mass_g',
               'RV_EDV_mL', 'RV_ESV_mL', 'RV_SV_mL', 'RV_EF_pct']
