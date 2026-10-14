"""Ventricular volumes and function from a 4D label map (x, y, slice, frame)."""
from __future__ import annotations

import numpy as np

from tigerhx.core.aha import LV, MYO, RV

MYOCARDIAL_DENSITY = 1.05  # g/mL

# How the ED frame is chosen when the user has not set it.
ED_RULES = ('max_volume', 'first_frame')


def voxel_ml(zooms):
    return float(np.prod(np.asarray(zooms[:3], dtype=float))) / 1000.0


def volume_curves(seg, zooms):
    """Volume (mL) of LV blood, myocardium and RV blood for every frame
    (all slices, no basal exclusion)."""
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


def included_slices(nz, first_slice, base_first):
    """Boolean mask of slices kept when the basal-most valid slice is
    first_slice (None keeps all): slices on its basal side are dropped."""
    z = np.arange(nz)
    if first_slice is None:
        return np.ones(nz, dtype=bool)
    return z >= first_slice if base_first else z <= first_slice


def _ef(edv, esv):
    return (edv - esv) / edv * 100.0 if edv > 0 else None


def compute_metrics(seg, zooms, ed, es, basal=None, base_first=True):
    """Clinical summary at the ED and ES frames, for LV and RV alike.

    basal: {'ED': {'LV': k, 'RV': k}, 'ES': {...}} with the basal-most slice
    to count for each phase and chamber (None or missing = all slices).
    LV mass uses the LV cutoff at ED.  EF/SV/ESV are None for single-frame
    data.
    """
    v = voxel_ml(zooms)
    nz = seg.shape[2]
    multi = seg.shape[3] > 1
    basal = basal or {}

    def vol(label, frame, phase, chamber):
        cut = (basal.get(phase) or {}).get(chamber)
        keep = included_slices(nz, cut, base_first)
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


METRIC_KEYS = ['LV_EDV_mL', 'LV_ESV_mL', 'LV_SV_mL', 'LV_EF_pct', 'LV_mass_g',
               'RV_EDV_mL', 'RV_ESV_mL', 'RV_SV_mL', 'RV_EF_pct']
