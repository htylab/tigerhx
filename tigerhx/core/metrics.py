"""Ventricular volumes and function from a 4D label map (x, y, slice, frame)."""
from __future__ import annotations

import numpy as np

from tigerhx.core.aha import LV, MYO, RV

MYOCARDIAL_DENSITY = 1.05  # g/mL


def voxel_ml(zooms):
    return float(np.prod(np.asarray(zooms[:3], dtype=float))) / 1000.0


def volume_curves(seg, zooms):
    """Volume (mL) of LV blood, myocardium and RV blood for every frame."""
    v = voxel_ml(zooms)
    return {name: ((seg == label).sum(axis=(0, 1, 2)) * v).tolist()
            for name, label in (('LV', LV), ('MYO', MYO), ('RV', RV))}


def find_ed_es(seg):
    """ED/ES frames as the max/min LV-blood voxel count (original GUI rule)."""
    curve = (seg == LV).sum(axis=(0, 1, 2))
    return int(np.argmax(curve)), int(np.argmin(curve))


def _ef(edv, esv):
    return (edv - esv) / edv * 100.0 if edv > 0 else None


def compute_metrics(seg, zooms, ed, es):
    """Clinical summary.  LV values use the given ED/ES frames; RV values use
    the max/min of the RV curve.  EF/SV are None for single-frame data."""
    curves = volume_curves(seg, zooms)
    lv, myo, rv = curves['LV'], curves['MYO'], curves['RV']
    multi = len(lv) > 1
    m = {
        'LV_EDV_mL': lv[ed],
        'LV_ESV_mL': lv[es] if multi else None,
        'LV_mass_g': myo[ed] * MYOCARDIAL_DENSITY,
        'RV_EDV_mL': max(rv),
        'RV_ESV_mL': min(rv) if multi else None,
    }
    m['LV_SV_mL'] = lv[ed] - lv[es] if multi else None
    m['LV_EF_pct'] = _ef(lv[ed], lv[es]) if multi else None
    m['RV_SV_mL'] = max(rv) - min(rv) if multi else None
    m['RV_EF_pct'] = _ef(max(rv), min(rv)) if multi else None
    return {k: (None if val is None else round(float(val), 2)) for k, val in m.items()}


METRIC_KEYS = ['LV_EDV_mL', 'LV_ESV_mL', 'LV_SV_mL', 'LV_EF_pct', 'LV_mass_g',
               'RV_EDV_mL', 'RV_ESV_mL', 'RV_SV_mL', 'RV_EF_pct']
