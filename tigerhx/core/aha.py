"""AHA myocardial segmentation on short-axis label maps.

Label convention of the segmentation: 1 = LV blood, 2 = LV myocardium,
3 = RV blood.  Arrays are indexed (x, y, slice, frame).

Sector convention (kept identical to the original tigerhx GUI): the angle
reference is the direction from the LV-blood centroid to the RV-blood
centroid, so the septum lies on the boundary between sectors 2 and 3 for
6-sector slices and in the middle of sector 2 for 4-sector slices.
"""
from __future__ import annotations

import numpy as np

LV, MYO, RV = 1, 2, 3


def _centroid(mask):
    ii, jj = np.nonzero(mask)
    return ii.mean(), jj.mean()


def rv_angle(slice_mask):
    """Angle (degrees) from the LV centroid to the RV centroid, or None."""
    lv = slice_mask == LV
    rv = slice_mask == RV
    if not lv.any() or not rv.any():
        return None
    lx, ly = _centroid(lv)
    rx, ry = _centroid(rv)
    return float(np.degrees(np.arctan2(-(rx - lx), ry - ly)))


def sector_map(slice_mask, nseg=6, ref_angle=None, center=None):
    """Assign every myocardium pixel of a 2D slice to sector 1..nseg.

    ref_angle defaults to the RV direction of this slice; center defaults to
    the LV-blood centroid.  Returns zeros when either cannot be determined.
    """
    if nseg not in (4, 6):
        raise ValueError('nseg must be 4 or 6')
    out = np.zeros(slice_mask.shape, dtype=np.int64)
    myo = slice_mask == MYO
    if not myo.any():
        return out
    if center is None:
        lv = slice_mask == LV
        if not lv.any():
            return out
        center = _centroid(lv)
    if ref_angle is None:
        ref_angle = rv_angle(slice_mask)
        if ref_angle is None:
            return out
    ii, jj = np.nonzero(myo)
    theta = np.degrees(np.arctan2(-(ii - center[0]), jj - center[1]))
    start = ref_angle - (120 if nseg == 6 else 135)
    sector = (np.mod(theta - start, 360) // (360 / nseg)).astype(np.int64) + 1
    out[ii, jj] = np.clip(sector, 1, nseg)
    return out


def _is_apical(z, apex_slice, base_first):
    if apex_slice is None:
        return False
    return z >= apex_slice if base_first else z <= apex_slice


def legacy_lvm(seg, apex_slice, base_first=True):
    """Myocardium sector map as written by the original GUI (field 'LVM').

    6 sectors before the apex slice and 4 from it on; slices missing any of
    the three labels keep the whole myocardium as 1.
    """
    lvm = np.zeros(seg.shape, dtype=np.int64)
    for z in range(seg.shape[2]):
        nseg = 4 if _is_apical(z, apex_slice, base_first) else 6
        for t in range(seg.shape[3]):
            m = seg[:, :, z, t]
            if (m == LV).any() and (m == MYO).any() and (m == RV).any():
                lvm[:, :, z, t] = sector_map(m, nseg)
            else:
                lvm[:, :, z, t] = m == MYO
    return lvm


def legacy_seg_aha(seg, lvm):
    """Original 'SegAHA' field: segmentation with sectors written as 8..13."""
    out = seg.astype(np.int64)
    out[lvm > 0] = lvm[lvm > 0] + 7
    return out


def _slice_area(seg, label, frame):
    return (seg[:, :, :, frame] == label).sum(axis=(0, 1))


def detect_base_first(seg, ed):
    """True when the slice index runs from base to apex.

    The LV cavity shrinks towards the apex, so the sign of a linear fit of
    the ED cavity area against slice index gives the stack direction.
    """
    area = _slice_area(seg, LV, ed)
    z = np.nonzero(area)[0]
    if z.size < 2:
        return True
    slope = np.polyfit(z, area[z].astype(float), 1)[0]
    return bool(slope <= 0)


def suggest_apex_slice(seg, ed, base_first):
    """First apical slice: start of the apical third of the slices with an
    LV cavity at ED.  None when no slice has a cavity."""
    area = _slice_area(seg, LV, ed)
    order = range(seg.shape[2]) if base_first else range(seg.shape[2] - 1, -1, -1)
    cavity = [z for z in order if area[z] > 0]
    if not cavity:
        return None
    n_apical = max(1, int(round(len(cavity) / 3)))
    return int(cavity[len(cavity) - n_apical])


def slice_levels(seg, ed, apex_slice, base_first):
    """Map each slice index to 'basal', 'mid', 'apical', 'apex' or None.

    Slices on the apical side of apex_slice are 'apical', or 'apex' (AHA
    segment 17) when they hold myocardium but no cavity at ED.  The remaining
    myocardial slices are split in half into basal and mid.
    """
    nz = seg.shape[2]
    myo = _slice_area(seg, MYO, ed) > 0
    cavity = _slice_area(seg, LV, ed) > 0
    order = list(range(nz)) if base_first else list(range(nz - 1, -1, -1))
    levels = {z: None for z in range(nz)}
    upper = []
    for z in order:
        if not myo[z]:
            continue
        if _is_apical(z, apex_slice, base_first):
            levels[z] = 'apical' if cavity[z] else 'apex'
        else:
            upper.append(z)
    n_basal = (len(upper) + 1) // 2
    for k, z in enumerate(upper):
        levels[z] = 'basal' if k < n_basal else 'mid'
    return levels


_LEVEL_OFFSET = {'basal': 0, 'mid': 6, 'apical': 12}


def aha17(seg, apex_slice, base_first, ed):
    """AHA 17-segment labels (1..17) on the myocardium, 0 elsewhere.

    Slices without an RV borrow the RV direction of the nearest slice of the
    same frame; slices without a cavity use the myocardium centroid.
    """
    out = np.zeros(seg.shape, dtype=np.uint8)
    levels = slice_levels(seg, ed, apex_slice, base_first)
    nz, nt = seg.shape[2], seg.shape[3]
    for t in range(nt):
        refs = {z: rv_angle(seg[:, :, z, t]) for z in range(nz)}
        known = [z for z in range(nz) if refs[z] is not None]
        for z in range(nz):
            level = levels[z]
            m = seg[:, :, z, t]
            myo = m == MYO
            if level is None or not myo.any():
                continue
            if level == 'apex':
                out[:, :, z, t][myo] = 17
                continue
            ref = refs[z]
            if ref is None:
                if not known:
                    continue
                ref = refs[min(known, key=lambda k: abs(k - z))]
            center = _centroid(m == LV) if (m == LV).any() else _centroid(myo)
            nseg = 4 if level == 'apical' else 6
            sec = sector_map(m, nseg, ref_angle=ref, center=center)
            out[:, :, z, t][sec > 0] = sec[sec > 0] + _LEVEL_OFFSET[level]
    return out
