//! Per-case processing: segmentation, analysis and output files.
//!
//! Segmentation is the only slow step and needs no user input.  The apex
//! choice only affects the AHA analysis, which is cheap to redo, so it is
//! decided after the user has seen the result.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use crate::error::{Error, Result};
use crate::inference::{model_version, segment};
use crate::legacy::{crop_box, legacy_result, strip_nii_ext};
use crate::mat::{load_mat, MatData, MatVar};
use crate::nifti::{self, Header};
use crate::project::{Case, Kind, Project, Status};
use crate::volume::Vol;
use crate::{aha, metrics};

pub fn seg_path(project: &Project, case: &Case) -> PathBuf {
    project.case_dir(case).join("seg.nii.gz")
}

pub fn aha_path(project: &Project, case: &Case) -> PathBuf {
    project.case_dir(case).join("aha17.nii.gz")
}

pub fn sectors_path(project: &Project, case: &Case) -> PathBuf {
    project.case_dir(case).join("sectors.nii.gz")
}

pub fn has_seg(project: &Project, case: &Case) -> bool {
    seg_path(project, case).is_file()
}

/// The case's 4D label map (x, y, slice, frame).
pub fn load_seg(project: &Project, case: &Case) -> Result<Vol<u8>> {
    nifti::load_labels(&seg_path(project, case))
}

fn get(project: &Project, id: &str) -> Result<Case> {
    project.case(id).ok_or_else(|| Error::not_found(format!("case {id} not found")))
}

/// Number of dimensions to write a label map with, given the source image.
fn label_ndim(src: &Header) -> usize {
    if src.ndim() == 4 { 4 } else { 3 }
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// An original-GUI result (*_pred_*.mat): image and labels as 4D volumes,
/// the LVM sector field, voxel size padded to 4 and model name.
pub struct LegacyMat {
    pub img: Vol<f32>,
    pub seg: Vol<u8>,
    pub lvm: Option<Vol<u8>>,
    pub zooms: [f64; 4],
    pub model: String,
}

fn shape4(v: &MatVar) -> Result<[usize; 4]> {
    if v.dims.len() > 4 {
        return Err(Error::invalid(format!("expected at most 4 dimensions, got {:?}", v.dims)));
    }
    let mut s = [1; 4];
    for (i, d) in v.dims.iter().enumerate() {
        s[i] = *d;
    }
    Ok(s)
}

fn nums(v: &MatVar) -> &[f64] {
    match &v.data {
        MatData::Num(n) => n,
        MatData::Str(_) => &[],
    }
}

pub fn load_legacy_mat(path: &Path) -> Result<LegacyMat> {
    let mut m = load_mat(path)?;
    for key in ["input_img", "Seg"] {
        if !matches!(m.get(key), Some(MatVar { data: MatData::Num(_), .. })) {
            return Err(Error::invalid(format!("{} has no {key}: not a TigerHx result", file_name(path))));
        }
    }
    let as_u8 = |v: &MatVar| -> Result<Vol<u8>> {
        // numpy's astype(uint8)
        Ok(Vol::from_vec(nums(v).iter().map(|&x| if x.is_finite() { (x as i64) as u8 } else { 0 }).collect(), shape4(v)?))
    };
    let img = m.remove("input_img").unwrap();
    let img = Vol::from_vec(nums(&img).iter().map(|&x| x as f32).collect(), shape4(&img)?);
    let seg = as_u8(&m["Seg"])?;
    let lvm = m.get("LVM").map(as_u8).transpose()?;
    let mut zooms = [1.0; 4];
    if let Some(v) = m.get("voxel_size") {
        for (i, z) in nums(v).iter().take(4).enumerate() {
            zooms[i] = *z;
        }
    }
    let model = match m.get("model").map(|v| &v.data) {
        Some(MatData::Str(s)) => s.trim_end().to_string(),
        _ => String::new(),
    };
    Ok(LegacyMat { img, seg, lvm, zooms, model })
}

/// The image of a case (always 4D) and its zooms; for an opened
/// original-GUI result the image stored in its .mat.
pub fn load_image(case: &Case) -> Result<(Vol<f32>, [f64; 4])> {
    if case.kind == Kind::Mat {
        let m = load_legacy_mat(Path::new(&case.path))?;
        return Ok((m.img, m.zooms));
    }
    let (data, header) = nifti::load(Path::new(&case.path))?;
    Ok((data, header.zooms4()))
}

/// Header and number of dimensions for label maps saved next to a case.
fn label_target(case: &Case) -> Result<(Header, usize)> {
    if case.kind == Kind::Mat {
        let z = case.zooms.clone().unwrap_or_default();
        let g = |i: usize| z.get(i).copied().unwrap_or(1.0);
        return Ok((nifti::plain_header([g(0), g(1), g(2)]), 4));
    }
    let src = nifti::read_header(Path::new(&case.path))?;
    let n = label_ndim(&src);
    Ok((src, n))
}

/// Open an original-GUI result: its labels become the case's segmentation,
/// its apex (read from the LVM sectors, base first as that GUI assumed)
/// counts as confirmed.  Nothing is written outside the workspace.
pub fn import_mat_case(project: &Project, id: &str) -> Result<()> {
    let case = get(project, id)?;
    let m = load_legacy_mat(Path::new(&case.path))?;
    std::fs::create_dir_all(project.case_dir(&case))?;
    let header = nifti::plain_header([m.zooms[0], m.zooms[1], m.zooms[2]]);
    nifti::save_labels(&m.seg, 4, &header, &seg_path(project, &case))?;
    let apex = m.lvm.as_ref().and_then(aha::legacy_apex_slice);
    project.update(id, |c| {
        c.model = m.model.clone();
        c.shape = Some(m.img.shape.to_vec());
        c.zooms = Some(m.zooms.to_vec());
        c.base_first = apex.map(|_| true);
        c.apex_slice = apex;
        c.apex_confirmed = apex.is_some();
    });
    analyze_case(project, id, Some(&m.seg)).map(|_| ())
}

/// Run the model, save the label map and analyse it, keeping an apex that is
/// already confirmed (from a CSV) and suggesting one otherwise.
pub fn segment_case(
    project: &Project,
    id: &str,
    model: &Path,
    progress: &dyn Fn(usize, usize),
    cancel: &AtomicBool,
) -> Result<Vol<u8>> {
    let case = get(project, id)?;
    let (data, header) = nifti::load(Path::new(&case.path))?;
    let seg = segment(&data, model, progress, cancel, 50)?;
    let out_dir = project.case_dir(&case);
    std::fs::create_dir_all(&out_dir)?;
    nifti::save_labels(&seg, label_ndim(&header), &header, &seg_path(project, &case))?;
    project.update(id, |c| {
        c.model = file_name(model);
        c.shape = Some(data.shape.to_vec());
        c.zooms = Some(header.zooms4().to_vec());
        c.base_first = None;
        if !(c.apex_confirmed && c.apex_slice.is_some()) {
            c.apex_slice = None;
            c.apex_confirmed = false;
        }
    });
    analyze_case(project, id, Some(&seg))?;
    write_outputs(project, id, Some((&data, &header)), Some(&seg))?;
    Ok(seg)
}

/// ED/ES frames, slice order, apex suggestion (unless already set), the
/// AHA17 map and the segmentation checks.  Fast enough to rerun whenever the
/// user changes the apex.
pub fn analyze_case(project: &Project, id: &str, seg: Option<&Vol<u8>>) -> Result<Vol<u8>> {
    let case = get(project, id)?;
    let loaded;
    let seg = match seg {
        Some(s) => s,
        None => {
            loaded = load_seg(project, &case)?;
            &loaded
        }
    };
    let (ed, es) = metrics::find_ed_es(seg);
    let mut apex = case.apex_slice;
    if apex.is_some_and(|a| a >= seg.nz()) {
        // e.g. a CSV apex out of range
        apex = None;
        project.update(id, |c| {
            c.apex_slice = None;
            c.apex_confirmed = false;
        });
    }
    let base_first = case.base_first.unwrap_or_else(|| aha::detect_base_first(seg, ed));
    let apex = apex.or_else(|| aha::suggest_apex_slice(seg, ed, base_first));
    let a17 = aha::aha17(seg, apex, base_first, ed);
    let sectors = aha::legacy_lvm(seg, apex, base_first);
    let (target, ndim) = label_target(&case)?;
    std::fs::create_dir_all(project.case_dir(&case))?;
    nifti::save_labels(&a17, ndim, &target, &aha_path(project, &case))?;
    nifti::save_labels(&sectors, ndim, &target, &sectors_path(project, &case))?;
    let levels = aha::slice_levels(seg, ed, apex, base_first);
    let checks = metrics::slice_checks(seg, ed, es, base_first, None, case.zooms.as_deref());
    let crop = crop_box(seg, 10);
    project.update(id, |c| {
        c.ed = Some(ed);
        c.es = Some(es);
        c.base_first = Some(base_first);
        c.apex_slice = apex;
        c.levels = Some(levels);
        c.checks = Some(checks);
        c.crop = Some(crop);
    });
    Ok(a17)
}

/// User-confirmed apex (and optionally slice order); reanalyse.
pub fn set_apex(project: &Project, id: &str, apex_slice: usize, base_first: Option<bool>) -> Result<()> {
    project.update(id, |c| {
        c.apex_slice = Some(apex_slice);
        if let Some(b) = base_first {
            c.base_first = Some(b);
        }
        c.apex_confirmed = true;
        c.status = Status::Done;
    });
    analyze_case(project, id, None).map(|_| ())
}

/// Name of the case's .mat in the output folder, as the original GUI named
/// it: the path relative to the added folder, joined with '_'.
pub fn mat_name(case: &Case) -> String {
    format!("{}_pred_{}.mat", case.name.replace('/', "_"), model_version(&case.model))
}

/// Legacy .mat (the output folder and/or the input folder) and input-folder
/// NIfTI.  A case from a CSV can override the two input-folder settings; an
/// opened original-GUI result writes nothing.
pub fn write_outputs(
    project: &Project,
    id: &str,
    image: Option<(&Vol<f32>, &Header)>,
    seg: Option<&Vol<u8>>,
) -> Result<()> {
    let case = get(project, id)?;
    if case.kind == Kind::Mat {
        return Ok(());
    }
    let mut s = project.settings();
    s.mat_in_inputdir = case.mat_in_inputdir.unwrap_or(s.mat_in_inputdir);
    s.nii_in_inputdir = case.nii_in_inputdir.unwrap_or(s.nii_in_inputdir);
    if !s.writes_outputs() {
        return Ok(());
    }
    let loaded_image;
    let (data, header) = match image {
        Some(i) => i,
        None => {
            loaded_image = nifti::load(Path::new(&case.path))?;
            (&loaded_image.0, &loaded_image.1)
        }
    };
    let loaded_seg;
    let seg = match seg {
        Some(s) => s,
        None => {
            loaded_seg = load_seg(project, &case)?;
            &loaded_seg
        }
    };
    let ver = model_version(&case.model);
    let stem = strip_nii_ext(&case.path);
    let input_dir = Path::new(&case.path).parent().map(Path::to_path_buf).unwrap_or_default();
    let mut outputs = Vec::new();
    if s.save_mat || s.mat_in_inputdir {
        let r = legacy_result(
            data,
            seg,
            case.apex_slice,
            case.base_first.unwrap_or(false),
            &case.model,
            &header.pixdim,
            case.ed,
            case.es,
        );
        if s.save_mat {
            let dir = project.output_dir();
            std::fs::create_dir_all(&dir)?;
            let p = dir.join(mat_name(&case));
            r.save(&p)?;
            outputs.push(("mat", p));
        }
        if s.mat_in_inputdir {
            let p = input_dir.join(format!("{stem}_{ver}.mat"));
            r.save(&p)?;
            outputs.push(("mat_inputdir", p));
        }
    }
    if s.nii_in_inputdir {
        let p = input_dir.join(format!("{stem}_{ver}_seg.nii.gz"));
        nifti::save_labels(seg, label_ndim(header), header, &p)?;
        outputs.push(("nii_inputdir", p));
    }
    project.update(id, |c| {
        for (k, p) in outputs {
            c.outputs.insert(k.to_string(), p.to_string_lossy().into_owned().into());
        }
    });
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VolumeKind {
    Image,
    Seg,
    Aha,
    Sectors,
}

impl VolumeKind {
    pub fn parse(s: &str) -> Option<VolumeKind> {
        match s {
            "image" => Some(VolumeKind::Image),
            "seg" => Some(VolumeKind::Seg),
            "aha" => Some(VolumeKind::Aha),
            "sectors" => Some(VolumeKind::Sectors),
            _ => None,
        }
    }
}

/// Display NIfTI bytes for the viewer: the image, the labels or the AHA
/// map; `crop` cuts them to the heart (the case's zoom box).
pub fn display_volume(project: &Project, case: &Case, kind: VolumeKind, crop: bool) -> Result<Vec<u8>> {
    let cut = |shape: [usize; 4]| case.crop.filter(|_| crop).map(|[x0, x1, y0, y1]| {
        [x0.min(shape[0]), x1.min(shape[0]), y0.min(shape[1]), y1.min(shape[1])]
    });
    if kind == VolumeKind::Image {
        let (mut data, zooms) = load_image(case)?;
        if let Some([x0, x1, y0, y1]) = cut(data.shape) {
            data = data.crop_xy(x0, x1, y0, y1);
        }
        return Ok(nifti::display_bytes(nifti::DisplayData::F32(&data.data), data.shape, zooms));
    }
    let path = match kind {
        VolumeKind::Aha => aha_path(project, case),
        VolumeKind::Sectors => sectors_path(project, case),
        _ => seg_path(project, case),
    };
    if !path.is_file() {
        return Err(Error::not_found(file_name(&path)));
    }
    let mut data = nifti::load_labels(&path)?;
    if let Some([x0, x1, y0, y1]) = cut(data.shape) {
        data = data.crop_xy(x0, x1, y0, y1);
    }
    let mut zooms = [1.0; 4];
    if let Some(z) = &case.zooms {
        for (i, v) in z.iter().take(4).enumerate() {
            zooms[i] = *v;
        }
    }
    Ok(nifti::display_bytes(nifti::DisplayData::U8(&data.data), data.shape, zooms))
}
