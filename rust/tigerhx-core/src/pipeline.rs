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
use crate::nifti::{self, Header};
use crate::project::{Case, Project, Status};
use crate::volume::Vol;
use crate::{aha, metrics};

pub fn seg_path(project: &Project, case: &Case) -> PathBuf {
    project.case_dir(case).join("seg.nii.gz")
}

pub fn aha_path(project: &Project, case: &Case) -> PathBuf {
    project.case_dir(case).join("aha17.nii.gz")
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

/// Run the model, save the label map and analyse it with a suggested apex.
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
        c.apex_slice = None;
        c.apex_confirmed = false;
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
    let base_first = case.base_first.unwrap_or_else(|| aha::detect_base_first(seg, ed));
    let apex = case.apex_slice.or_else(|| aha::suggest_apex_slice(seg, ed, base_first));
    let a17 = aha::aha17(seg, apex, base_first, ed);
    let src = nifti::read_header(Path::new(&case.path))?;
    std::fs::create_dir_all(project.case_dir(&case))?;
    nifti::save_labels(&a17, label_ndim(&src), &src, &aha_path(project, &case))?;
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

/// Legacy .mat (workspace and/or input folder) and input-folder NIfTI,
/// according to the project settings.
pub fn write_outputs(
    project: &Project,
    id: &str,
    image: Option<(&Vol<f32>, &Header)>,
    seg: Option<&Vol<u8>>,
) -> Result<()> {
    let s = project.settings();
    if !s.writes_outputs() {
        return Ok(());
    }
    let case = get(project, id)?;
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
            let p = project.case_dir(&case).join(format!("{stem}_pred_{ver}.mat"));
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
}

impl VolumeKind {
    pub fn parse(s: &str) -> Option<VolumeKind> {
        match s {
            "image" => Some(VolumeKind::Image),
            "seg" => Some(VolumeKind::Seg),
            "aha" => Some(VolumeKind::Aha),
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
        let (mut data, header) = nifti::load(Path::new(&case.path))?;
        if let Some([x0, x1, y0, y1]) = cut(data.shape) {
            data = data.crop_xy(x0, x1, y0, y1);
        }
        return Ok(nifti::display_bytes(nifti::DisplayData::F32(&data.data), data.shape, header.zooms4()));
    }
    let path = if kind == VolumeKind::Aha { aha_path(project, case) } else { seg_path(project, case) };
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
