//! AHA myocardial segmentation on short-axis label maps.
//!
//! Label convention of the segmentation: 1 = LV blood, 2 = LV myocardium,
//! 3 = RV blood.  Arrays are indexed (x, y, slice, frame).
//!
//! Sector convention (kept identical to the original tigerhx GUI): the angle
//! reference is the direction from the LV-blood centroid to the RV-blood
//! centroid, so the septum lies on the boundary between sectors 2 and 3 for
//! 6-sector slices and in the middle of sector 2 for 4-sector slices.

use serde::{Deserialize, Serialize};

use crate::volume::{Slice2, Vol};

pub const LV: u8 = 1;
pub const MYO: u8 = 2;
pub const RV: u8 = 3;

/// Mean (x, y) of the pixels passing `test`, or None if there are none.
pub fn centroid(sl: Slice2<'_, u8>, test: impl Fn(u8) -> bool) -> Option<(f64, f64)> {
    let (mut si, mut sj, mut n) = (0.0f64, 0.0f64, 0usize);
    for j in 0..sl.ny {
        for i in 0..sl.nx {
            if test(sl.get(i, j)) {
                si += i as f64;
                sj += j as f64;
                n += 1;
            }
        }
    }
    (n > 0).then(|| (si / n as f64, sj / n as f64))
}

fn any(sl: Slice2<'_, u8>, label: u8) -> bool {
    sl.data.contains(&label)
}

/// Angle (degrees) from the LV centroid to the RV centroid, or None.
pub fn rv_angle(sl: Slice2<'_, u8>) -> Option<f64> {
    let (lx, ly) = centroid(sl, |v| v == LV)?;
    let (rx, ry) = centroid(sl, |v| v == RV)?;
    Some((-(rx - lx)).atan2(ry - ly).to_degrees())
}

/// Assign every myocardium pixel of a 2D slice to sector 1..nseg.
///
/// ref_angle defaults to the RV direction of this slice; center defaults to
/// the LV-blood centroid.  Returns zeros when either cannot be determined.
pub fn sector_map(sl: Slice2<'_, u8>, nseg: u8, ref_angle: Option<f64>, center: Option<(f64, f64)>) -> Vec<u8> {
    assert!(nseg == 4 || nseg == 6, "nseg must be 4 or 6");
    let mut out = vec![0u8; sl.data.len()];
    if !any(sl, MYO) {
        return out;
    }
    let Some(center) = center.or_else(|| centroid(sl, |v| v == LV)) else { return out };
    let Some(ref_angle) = ref_angle.or_else(|| rv_angle(sl)) else { return out };
    let start = ref_angle - if nseg == 6 { 120.0 } else { 135.0 };
    let width = 360.0 / nseg as f64;
    for j in 0..sl.ny {
        for i in 0..sl.nx {
            if sl.get(i, j) != MYO {
                continue;
            }
            let theta = (-(i as f64 - center.0)).atan2(j as f64 - center.1).to_degrees();
            let sector = ((theta - start).rem_euclid(360.0) / width).floor() as i64 + 1;
            out[i + sl.nx * j] = sector.clamp(1, nseg as i64) as u8;
        }
    }
    out
}

fn is_apical(z: usize, apex_slice: Option<usize>, base_first: bool) -> bool {
    match apex_slice {
        None => false,
        Some(a) => if base_first { z >= a } else { z <= a },
    }
}

/// Myocardium sector map as written by the original GUI (field 'LVM').
///
/// 6 sectors before the apex slice and 4 from it on; slices missing any of
/// the three labels keep the whole myocardium as 1.
pub fn legacy_lvm(seg: &Vol<u8>, apex_slice: Option<usize>, base_first: bool) -> Vol<u8> {
    let mut lvm = Vol::<u8>::zeros(seg.shape);
    for z in 0..seg.nz() {
        let nseg = if is_apical(z, apex_slice, base_first) { 4 } else { 6 };
        for t in 0..seg.nt() {
            let m = seg.slice2(z, t);
            let out = if any(m, LV) && any(m, MYO) && any(m, RV) {
                sector_map(m, nseg, None, None)
            } else {
                m.data.iter().map(|&v| (v == MYO) as u8).collect()
            };
            lvm.slice_mut(z, t).copy_from_slice(&out);
        }
    }
    lvm
}

/// Original 'SegAHA' field: segmentation with sectors written as 8..13.
pub fn legacy_seg_aha(seg: &Vol<u8>, lvm: &Vol<u8>) -> Vol<u8> {
    let data = seg.data.iter().zip(&lvm.data).map(|(&s, &l)| if l > 0 { l + 7 } else { s }).collect();
    Vol::from_vec(data, seg.shape)
}

/// Pixel count of a label on every slice of one frame.
pub fn slice_counts(seg: &Vol<u8>, label: u8, frame: usize) -> Vec<usize> {
    (0..seg.nz()).map(|z| seg.slice(z, frame).iter().filter(|&&v| v == label).count()).collect()
}

/// True when the slice index runs from base to apex.
///
/// The LV cavity shrinks towards the apex, so the sign of a linear fit of
/// the ED cavity area against slice index gives the stack direction.
pub fn detect_base_first(seg: &Vol<u8>, ed: usize) -> bool {
    let area = slice_counts(seg, LV, ed);
    let pts: Vec<(f64, f64)> =
        area.iter().enumerate().filter(|(_, &a)| a > 0).map(|(z, &a)| (z as f64, a as f64)).collect();
    if pts.len() < 2 {
        return true;
    }
    let n = pts.len() as f64;
    let mz = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let ma = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let cov: f64 = pts.iter().map(|p| (p.0 - mz) * (p.1 - ma)).sum();
    cov <= 0.0 // the slope has the sign of the covariance
}

fn order(nz: usize, base_first: bool) -> Vec<usize> {
    if base_first { (0..nz).collect() } else { (0..nz).rev().collect() }
}

/// First apical slice: start of the apical third of the slices with an LV
/// cavity at ED.  None when no slice has a cavity.
pub fn suggest_apex_slice(seg: &Vol<u8>, ed: usize, base_first: bool) -> Option<usize> {
    let area = slice_counts(seg, LV, ed);
    let cavity: Vec<usize> = order(seg.nz(), base_first).into_iter().filter(|&z| area[z] > 0).collect();
    if cavity.is_empty() {
        return None;
    }
    let n_apical = ((cavity.len() as f64 / 3.0).round() as usize).max(1);
    Some(cavity[cavity.len() - n_apical])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Basal,
    Mid,
    Apical,
    /// AHA segment 17: myocardium without a cavity beyond the apical start.
    Apex,
}

/// The level of each slice index, or None for slices without myocardium.
///
/// Slices on the apical side of apex_slice are 'apical', or 'apex' (AHA
/// segment 17) when they hold myocardium but no cavity at ED.  The remaining
/// myocardial slices are split in half into basal and mid.
pub fn slice_levels(seg: &Vol<u8>, ed: usize, apex_slice: Option<usize>, base_first: bool) -> Vec<Option<Level>> {
    let nz = seg.nz();
    let myo = slice_counts(seg, MYO, ed);
    let cavity = slice_counts(seg, LV, ed);
    let mut levels = vec![None; nz];
    let mut upper = Vec::new();
    for z in order(nz, base_first) {
        if myo[z] == 0 {
            continue;
        }
        if is_apical(z, apex_slice, base_first) {
            levels[z] = Some(if cavity[z] > 0 { Level::Apical } else { Level::Apex });
        } else {
            upper.push(z);
        }
    }
    let n_basal = upper.len().div_ceil(2);
    for (k, &z) in upper.iter().enumerate() {
        levels[z] = Some(if k < n_basal { Level::Basal } else { Level::Mid });
    }
    levels
}

/// AHA 17-segment labels (1..17) on the myocardium, 0 elsewhere.
///
/// Slices without an RV borrow the RV direction of the nearest slice of the
/// same frame; slices without a cavity use the myocardium centroid.
pub fn aha17(seg: &Vol<u8>, apex_slice: Option<usize>, base_first: bool, ed: usize) -> Vol<u8> {
    let mut out = Vol::<u8>::zeros(seg.shape);
    let levels = slice_levels(seg, ed, apex_slice, base_first);
    let (nz, nt) = (seg.nz(), seg.nt());
    for t in 0..nt {
        let refs: Vec<Option<f64>> = (0..nz).map(|z| rv_angle(seg.slice2(z, t))).collect();
        let known: Vec<usize> = (0..nz).filter(|&z| refs[z].is_some()).collect();
        for z in 0..nz {
            let m = seg.slice2(z, t);
            let Some(level) = levels[z] else { continue };
            if !any(m, MYO) {
                continue;
            }
            let dst = out.slice_mut(z, t);
            if level == Level::Apex {
                for (o, &v) in dst.iter_mut().zip(m.data) {
                    if v == MYO {
                        *o = 17;
                    }
                }
                continue;
            }
            let reference = match refs[z] {
                Some(r) => r,
                None => match known.iter().min_by_key(|&&k| k.abs_diff(z)) {
                    Some(&k) => refs[k].unwrap(),
                    None => continue,
                },
            };
            let center = centroid(m, |v| v == LV).or_else(|| centroid(m, |v| v == MYO));
            let (nseg, offset) = match level {
                Level::Basal => (6, 0),
                Level::Mid => (6, 6),
                _ => (4, 12),
            };
            let sec = sector_map(m, nseg, Some(reference), center);
            for (o, &s) in dst.iter_mut().zip(&sec) {
                if s > 0 {
                    *o = s + offset;
                }
            }
        }
    }
    out
}
