//! Names, the zoom box and the legacy .mat result of the original GUI.

use std::path::Path;

use crate::aha::{self, LV, RV};
use crate::error::Result;
use crate::mat::{save_mat, Data, Var};
use crate::volume::Vol;

/// 'a.b.cine.nii.gz' -> 'a.b.cine' (only the NIfTI extension is removed).
pub fn strip_nii_ext(path: &str) -> String {
    let name = Path::new(path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let lower = name.to_lowercase();
    for ext in [".nii.gz", ".nii"] {
        if lower.ends_with(ext) {
            return name[..name.len() - ext.len()].to_string();
        }
    }
    name
}

/// Runs of characters other than word characters, '.' and '-' become '_'.
pub fn safe_name(text: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"[^\w.\-]+").unwrap());
    let s = re.replace_all(text, "_");
    let s = s.trim_matches(|c| c == '.' || c == '_');
    if s.is_empty() { "case".into() } else { s.into() }
}

/// [x0, x1, y0, y1] around every label in any slice and frame, plus a margin
/// in pixels (the original GUI's '_crop' views); the full image when there
/// is no label.
pub fn crop_box(seg: &Vol<u8>, margin: usize) -> [usize; 4] {
    let (nx, ny) = (seg.nx(), seg.ny());
    let mut mask = vec![false; nx * ny];
    for chunk in seg.data.chunks_exact(nx * ny) {
        for (m, &v) in mask.iter_mut().zip(chunk) {
            *m |= v != 0;
        }
    }
    let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
    for j in 0..ny {
        for i in 0..nx {
            if mask[i + nx * j] {
                x0 = x0.min(i);
                x1 = x1.max(i);
                y0 = y0.min(j);
                y1 = y1.max(j);
            }
        }
    }
    if x0 == usize::MAX {
        return [0, nx, 0, ny];
    }
    [x0.saturating_sub(margin), (x1 + 1 + margin).min(nx), y0.saturating_sub(margin), (y1 + 1 + margin).min(ny)]
}

/// The result written by the original GUI, field for field, plus 'AHA17'.
pub struct LegacyResult<'a> {
    img: &'a Vol<f32>,
    seg: &'a Vol<u8>,
    seg_aha: Vol<u8>,
    lvm: Vol<u8>,
    lv: Vol<u8>,
    rv: Vol<u8>,
    aha17: Vol<u8>,
    crop: [usize; 4],
    voxel_size: Vec<f64>,
    model: String,
    /// (ed, es) for multi-frame data
    phases: Option<(usize, usize)>,
}

/// img and seg are 4D; voxel_size as header.get_zooms(); ed/es default to the original rule (max/min LV-blood
/// voxel count).
#[allow(clippy::too_many_arguments)]
pub fn legacy_result<'a>(
    img: &'a Vol<f32>,
    seg: &'a Vol<u8>,
    apex_slice: Option<usize>,
    base_first: bool,
    model: &str,
    voxel_size: &[f64],
    ed: Option<usize>,
    es: Option<usize>,
) -> LegacyResult<'a> {
    let lvm = aha::legacy_lvm(seg, apex_slice, base_first);
    let seg_aha = aha::legacy_seg_aha(seg, &lvm);
    // the original crop: 10 px around the labels, end index exclusive of +10
    let (nx, ny) = (seg.nx(), seg.ny());
    let crop = if seg.data.iter().any(|&v| v != 0) {
        let [xa, xb, ya, yb] = crop_box(seg, 0); // xb, yb: last index + 1
        [xa.saturating_sub(10), (xb - 1 + 10).min(nx), ya.saturating_sub(10), (yb - 1 + 10).min(ny)]
    } else {
        [0, nx, 0, ny]
    };
    let phases = if seg.nt() > 1 {
        let (red, res) = crate::metrics::find_ed_es(seg);
        Some((ed.unwrap_or(red), es.unwrap_or(res)))
    } else {
        None
    };
    let aha_ed = phases.map(|p| p.0).unwrap_or(0);
    LegacyResult {
        img,
        lv: seg.map(|v| (v == LV) as u8),
        rv: seg.map(|v| (v == RV) as u8),
        aha17: aha::aha17(seg, apex_slice, base_first, aha_ed),
        seg,
        seg_aha,
        lvm,
        crop,
        voxel_size: voxel_size.to_vec(),
        model: Path::new(model).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        phases,
    }
}

impl LegacyResult<'_> {
    /// The zooms padded to 4 values: float32 as nibabel gives them, but
    /// float64 once padded (numpy promotes the Python 1.0).
    fn voxel_size_var(&self) -> Var<'static> {
        let mut v = self.voxel_size.clone();
        let padded = v.len() < 4;
        v.resize(4, 1.0);
        v.truncate(4);
        if padded {
            Var::new("voxel_size", &[4], Data::F64(v))
        } else {
            Var::new("voxel_size", &[4], Data::F32(v.iter().map(|&x| x as f32).collect()))
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let [x0, x1, y0, y1] = self.crop;
        let img_crop = self.img.crop_xy(x0, x1, y0, y1);
        let seg_crop = self.seg.crop_xy(x0, x1, y0, y1);
        let aha_crop = self.seg_aha.crop_xy(x0, x1, y0, y1);
        let dims = |v: &Vol<u8>| v.shape.to_vec();
        let mut vars = vec![
            Var::new("input_img", &self.img.shape, Data::F32AsDouble(&self.img.data)),
            Var::new("Seg", &self.seg.shape, Data::U8AsInt64(&self.seg.data)),
            Var::new("SegAHA", &self.seg_aha.shape, Data::U8AsInt64(&self.seg_aha.data)),
            Var::new("input_crop", &img_crop.shape, Data::F32AsDouble(&img_crop.data)),
            Var::new("Seg_crop", &dims(&seg_crop), Data::U8AsInt64(&seg_crop.data)),
            Var::new("SegAHA_crop", &dims(&aha_crop), Data::U8AsInt64(&aha_crop.data)),
            Var::new("LV", &dims(&self.lv), Data::U8AsInt64(&self.lv.data)),
            Var::new("LVM", &dims(&self.lvm), Data::U8AsInt64(&self.lvm.data)),
            Var::new("RV", &dims(&self.rv), Data::U8AsInt64(&self.rv.data)),
            self.voxel_size_var(),
            Var::string("model", &self.model),
        ];
        let frames;
        if let Some((ed, es)) = self.phases {
            frames = [ed, es].map(|t| (self.seg.frame_vol(t), self.seg_aha.frame_vol(t)));
            let d3 = self.seg.shape[..3].to_vec();
            let [(seg_ed, aha_ed), (seg_es, aha_es)] = &frames;
            vars.extend([
                Var::scalar_i64("ES_frame_0base", es as i64),
                Var::scalar_i64("ED_frame_0base", ed as i64),
                Var::new("Seg_ES", &d3, Data::U8AsInt64(&seg_es.data)),
                Var::new("Seg_ED", &d3, Data::U8AsInt64(&seg_ed.data)),
                Var::new("SegAHA_ES", &d3, Data::U8AsInt64(&aha_es.data)),
                Var::new("SegAHA_ED", &d3, Data::U8AsInt64(&aha_ed.data)),
            ]);
        }
        vars.push(Var::new("AHA17", &dims(&self.aha17), Data::U8AsInt64(&self.aha17.data)));
        save_mat(path, &vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(strip_nii_ext("/x/a.b.cine.nii.gz"), "a.b.cine");
        assert_eq!(strip_nii_ext("A.NII"), "A");
        assert_eq!(strip_nii_ext("a.txt"), "a.txt");
        assert_eq!(safe_name("sub 01/cine (1)"), "sub_01_cine_1");
        assert_eq!(safe_name("..."), "case");
        assert_eq!(safe_name("病例 一"), "病例_一");
    }

    #[test]
    fn crop() {
        let mut seg = Vol::<u8>::zeros([30, 40, 2, 2]);
        assert_eq!(crop_box(&seg, 10), [0, 30, 0, 40]);
        let i = seg.idx(12, 5, 1, 1);
        seg.data[i] = 2;
        assert_eq!(crop_box(&seg, 10), [2, 23, 0, 16]);
    }
}
