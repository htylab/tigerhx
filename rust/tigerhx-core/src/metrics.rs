//! ED/ES frames and the segmentation checks shown by the GUI.

use serde::{Deserialize, Serialize};

use crate::aha::{slice_counts, LV, MYO, RV};
use crate::volume::{Slice2, Vol};

pub const CHAMBERS: [(&str, u8); 2] = [("LV", LV), ("RV", RV)];

fn lv_curve(seg: &Vol<u8>) -> Vec<usize> {
    (0..seg.nt()).map(|t| seg.frame(t).iter().filter(|&&v| v == LV).count()).collect()
}

/// ED/ES frames: ED has the largest LV-blood volume, ES the smallest (the
/// first such frame on ties, like numpy's argmax/argmin).
pub fn find_ed_es(seg: &Vol<u8>) -> (usize, usize) {
    let curve = lv_curve(seg);
    let mut ed = 0;
    let mut es = 0;
    for (t, &v) in curve.iter().enumerate() {
        if v > curve[ed] {
            ed = t;
        }
        if v < curve[es] {
            es = t;
        }
    }
    (ed, es)
}

fn pixel_mm2(zooms: Option<&[f64]>) -> f64 {
    match zooms {
        Some(z) if z.len() >= 2 => z[0] * z[1],
        _ => 1.0,
    }
}

/// Area (mm^2) of a label on every slice of one frame.
fn areas(seg: &Vol<u8>, label: u8, frame: usize, px: f64) -> Vec<f64> {
    slice_counts(seg, label, frame).into_iter().map(|n| n as f64 * px).collect()
}

/// Python's int(round(x)) (ties to even).
fn round_int(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Fraction of the LV-blood border that touches myocardium (None for a
/// tiny LV).  Close to 1 inside the ventricle; an open ring at the base
/// suggests the slice is already atrium.
pub fn lv_ring_coverage(sl: Slice2<'_, u8>) -> Option<f64> {
    let (nx, ny) = (sl.nx, sl.ny);
    let lv = |i: isize, j: isize| -> bool {
        i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny && sl.get(i as usize, j as usize) == LV
    };
    let myo = |i: isize, j: isize| -> bool {
        i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny && sl.get(i as usize, j as usize) == MYO
    };
    if sl.data.iter().filter(|&&v| v == LV).count() < 20 {
        return None;
    }
    let (mut edge, mut touching) = (0usize, 0usize);
    for j in 0..ny as isize {
        for i in 0..nx as isize {
            if !lv(i, j) {
                continue;
            }
            // binary_erosion with the cross structure; outside the image is background
            let interior = lv(i - 1, j) && lv(i + 1, j) && lv(i, j - 1) && lv(i, j + 1);
            if interior {
                continue;
            }
            edge += 1;
            if myo(i, j) || myo(i - 1, j) || myo(i + 1, j) || myo(i, j - 1) || myo(i, j + 1) {
                touching += 1;
            }
        }
    }
    Some(touching as f64 / edge as f64)
}

/// A slice whose segmentation is worth a look (field names as in project.json).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Check {
    pub kind: String,
    pub phase: String,
    pub chamber: String,
    pub slice: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub es: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neighbours: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<f64>,
}

impl Check {
    fn new(kind: &str, phase: &str, chamber: &str, slice: usize) -> Self {
        Check {
            kind: kind.into(),
            phase: phase.into(),
            chamber: chamber.into(),
            slice,
            ed: None,
            es: None,
            area: None,
            neighbours: None,
            coverage: None,
        }
    }
}

/// Slices left out of a phase and chamber (by slice index).
pub type Exclusions = std::collections::HashMap<(String, String), Vec<usize>>;

/// Slices whose segmentation is worth a look.
///
/// 'es_gt_ed':  a cavity is clearly larger at ES than at ED on that slice
///              (often a structure outside the ventricle near the apex).
/// 'gap':       a cavity is (almost) missing on a slice although both
///              neighbours have it.
/// 'open_ring': the basal-most LV slice at ES is poorly enclosed by
///              myocardium, so the LV label probably covers atrium.
/// Neighbours listed in excluded do not count as evidence of a gap.
pub fn slice_checks(
    seg: &Vol<u8>,
    ed: usize,
    es: usize,
    base_first: bool,
    excluded: Option<&Exclusions>,
    zooms: Option<&[f64]>,
) -> Vec<Check> {
    let mut checks = Vec::new();
    if seg.nt() < 2 || ed == es {
        return checks;
    }
    let nz = seg.nz();
    let px = pixel_mm2(zooms);
    let min_area = 100.0; // mm^2
    for (chamber, label) in CHAMBERS {
        let a_ed = areas(seg, label, ed, px);
        let a_es = areas(seg, label, es, px);
        for z in 0..nz {
            if a_es[z] > 1.1 * a_ed[z] + 50.0 {
                let mut c = Check::new("es_gt_ed", "ES", chamber, z);
                c.ed = Some(round_int(a_ed[z]));
                c.es = Some(round_int(a_es[z]));
                checks.push(c);
            }
        }
        for (ph, a) in [("ED", &a_ed), ("ES", &a_es)] {
            let skip = excluded.and_then(|e| e.get(&(ph.to_string(), chamber.to_string())));
            for z in 1..nz.saturating_sub(1) {
                if skip.is_some_and(|s| s.contains(&(z - 1)) || s.contains(&(z + 1))) {
                    continue;
                }
                let lo = a[z - 1].min(a[z + 1]);
                if lo > min_area && a[z] < 0.2 * lo {
                    let mut c = Check::new("gap", ph, chamber, z);
                    c.area = Some(round_int(a[z]));
                    c.neighbours = Some(round_int(lo));
                    checks.push(c);
                }
            }
        }
        if chamber != "LV" {
            continue;
        }
        let mut order: Box<dyn Iterator<Item = usize>> =
            if base_first { Box::new(0..nz) } else { Box::new((0..nz).rev()) };
        let basal = order.find(|&z| a_es[z] > min_area);
        let cov = basal.and_then(|z| lv_ring_coverage(seg.slice2(z, es)));
        if let (Some(z), Some(cov)) = (basal, cov) {
            if cov < 0.75 {
                let mut c = Check::new("open_ring", "ES", chamber, z);
                c.coverage = Some((cov * 100.0).round_ties_even() / 100.0);
                checks.push(c);
            }
        }
    }
    checks
}
