"""Ventricular volumes and function from a 4D label map (x, y, slice, frame)."""
from __future__ import annotations

import numpy as np

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


def slice_checks(seg, ed, es, base_first=True, excluded=None, min_voxels=100):
    """Slices worth a look before trusting ED/ES volumes.

    'es_gt_ed': a cavity is clearly larger at ES than at ED on that slice
                (often a structure outside the ventricle near the apex).
    'gap':      a cavity is (almost) missing on a slice although both
                neighbours have it.
    'basal':    reminder to check that the basal-most slice with a cavity at
                ES is still ventricle and not atrium (the commonest source of
                ESV error, which these numbers alone cannot detect).
    Neighbours the user has excluded do not count as evidence of a gap.
    """
    excluded = excluded or {}
    checks = []
    if seg.shape[3] < 2 or ed == es:
        return checks
    nz = seg.shape[2]
    for chamber, label in CHAMBER_LABELS.items():
        a = {ph: (seg[..., f] == label).sum(axis=(0, 1)) for ph, f in (('ED', ed), ('ES', es))}
        for z in range(nz):
            if a['ES'][z] > 1.1 * a['ED'][z] + 50:
                checks.append({'kind': 'es_gt_ed', 'phase': 'ES', 'chamber': chamber, 'slice': z,
                               'ed': int(a['ED'][z]), 'es': int(a['ES'][z])})
        for ph in ('ED', 'ES'):
            skip = set((excluded.get(ph) or {}).get(chamber) or ())
            for z in range(1, nz - 1):
                if skip & {z - 1, z + 1}:
                    continue
                lo = min(a[ph][z - 1], a[ph][z + 1])
                if lo > min_voxels and a[ph][z] < 0.2 * lo:
                    checks.append({'kind': 'gap', 'phase': ph, 'chamber': chamber, 'slice': z,
                                   'area': int(a[ph][z]), 'neighbours': int(lo)})
        order = range(nz) if base_first else range(nz - 1, -1, -1)
        basal = next((z for z in order if a['ES'][z] > min_voxels), None)
        if basal is not None:
            checks.append({'kind': 'basal', 'phase': 'ES', 'chamber': chamber, 'slice': basal,
                           'es': int(a['ES'][basal])})
    return checks


METRIC_KEYS = ['LV_EDV_mL', 'LV_ESV_mL', 'LV_SV_mL', 'LV_EF_pct', 'LV_mass_g',
               'RV_EDV_mL', 'RV_ESV_mL', 'RV_SV_mL', 'RV_EF_pct']
