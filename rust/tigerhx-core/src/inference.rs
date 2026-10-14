//! ONNX inference for cine short-axis segmentation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ort::environment::Environment;
use ort::logging::LogLevel;
use ort::session::Session;
use ort::value::{TensorElementType, Tensor};

use crate::error::{Error, Result};
use crate::volume::Vol;

fn file_name(path: &Path) -> String {
    path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// 'xy' (slice-wise) or 'xyz' (volume-wise) from cine4d_vNNNN_<mode>_*.onnx.
pub fn model_mode(model: &Path) -> &'static str {
    match file_name(model).split('_').nth(2) {
        Some("xyz") => "xyz",
        _ => "xy",
    }
}

/// 'v0006' from cine4d_v0006_xy_cine.onnx.
pub fn model_version(model: &str) -> String {
    let name = file_name(Path::new(model));
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() > 1 { parts[1].to_string() } else { name }
}

/// Keep only the largest face-connected component of each label; a label
/// whose largest component is smaller than min_size voxels is removed.
///
/// `labels` is in C order for `shape` (last axis fastest).  Components are
/// numbered in raster order like scipy.ndimage.label, so ties keep the
/// component met first.
pub fn keep_largest_component(labels: &mut [u8], shape: &[usize], min_size: usize) {
    let n = labels.len();
    debug_assert_eq!(n, shape.iter().product::<usize>());
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    const NONE: u32 = u32::MAX;
    let mut comp = vec![NONE; n];
    // per label value: (component id, size) of the largest so far
    let mut best: HashMap<u8, (u32, usize)> = HashMap::new();
    let mut stack = Vec::new();
    let mut next_id = 0u32;
    for start in 0..n {
        let v = labels[start];
        if v == 0 || comp[start] != NONE {
            continue;
        }
        let id = next_id;
        next_id += 1;
        comp[start] = id;
        stack.push(start);
        let mut size = 0usize;
        while let Some(p) = stack.pop() {
            size += 1;
            let mut rem = p;
            for d in 0..shape.len() {
                let coord = rem / strides[d];
                rem %= strides[d];
                if coord > 0 {
                    let q = p - strides[d];
                    if labels[q] == v && comp[q] == NONE {
                        comp[q] = id;
                        stack.push(q);
                    }
                }
                if coord + 1 < shape[d] {
                    let q = p + strides[d];
                    if labels[q] == v && comp[q] == NONE {
                        comp[q] = id;
                        stack.push(q);
                    }
                }
            }
        }
        let e = best.entry(v).or_insert((id, size));
        if size > e.1 {
            *e = (id, size);
        }
    }
    for (l, c) in labels.iter_mut().zip(&comp) {
        if *l == 0 {
            continue;
        }
        let (id, size) = best[l];
        if *c != id || size < min_size {
            *l = 0;
        }
    }
}

fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("tigerhx").build()?;
    Ok(ENV.get_or_init(|| env))
}

type SharedSession = Arc<Mutex<Session>>;

/// Cached ONNX Runtime session (CPU).
pub fn get_session(model: &Path) -> Result<SharedSession> {
    static SESSIONS: OnceLock<Mutex<HashMap<PathBuf, SharedSession>>> = OnceLock::new();
    let cache = SESSIONS.get_or_init(Default::default);
    let mut map = cache.lock().unwrap();
    if let Some(s) = map.get(model) {
        return Ok(s.clone());
    }
    let session = Session::builder(environment()?)?
        .with_log_level(LogLevel::Error)
        .map_err(ort::Error::from)?
        .commit_from_file(model)
        .map_err(|e| Error::other(format!("cannot load model {}: {e}", model.display())))?;
    let s = Arc::new(Mutex::new(session));
    map.insert(model.to_path_buf(), s.clone());
    Ok(s)
}

/// Run the model on one C-order image of `dims` (without the batch and
/// channel axes); returns the per-voxel argmax, C order.
fn predict(session: &mut Session, image: &[f32], dims: &[usize]) -> Result<Vec<u8>> {
    let input = &session.inputs()[0];
    let name = input.name().to_string();
    let double = input.dtype().tensor_type() == Some(TensorElementType::Float64);
    let mut shape = vec![1i64, 1];
    shape.extend(dims.iter().map(|&d| d as i64));
    let outputs = if double {
        let x: Vec<f64> = image.iter().map(|&v| v as f64).collect();
        session.run(ort::inputs![name.as_str() => Tensor::from_array((shape, x))?])?
    } else {
        session.run(ort::inputs![name.as_str() => Tensor::from_array((shape, image.to_vec()))?])?
    };
    let n = image.len();
    let mut pred = vec![0u8; n];
    let out = &outputs[0];
    let mut argmax = |logits: &[f64], nc: usize| {
        for (p, out) in pred.iter_mut().enumerate() {
            let mut best = 0;
            for c in 1..nc {
                if logits[c * n + p] > logits[best * n + p] {
                    best = c;
                }
            }
            *out = best as u8;
        }
    };
    let (oshape, logits): (Vec<i64>, Vec<f64>) = match out.try_extract_tensor::<f32>() {
        Ok((s, v)) => (s.to_vec(), v.iter().map(|&x| x as f64).collect()),
        Err(_) => {
            let (s, v) = out.try_extract_tensor::<f64>()?;
            (s.to_vec(), v.to_vec())
        }
    };
    if oshape.len() < 2 || logits.len() != oshape[1] as usize * n {
        return Err(Error::other(format!("unexpected model output shape {oshape:?}")));
    }
    argmax(&logits, oshape[1] as usize);
    Ok(pred)
}

/// Segment a 4D (x, y, slice, frame) cine image.
///
/// `progress(done, total)` is called after every model call; setting
/// `cancel` stops the run with a Cancelled error.  Returns uint8 labels with
/// the same shape as img.
pub fn segment(
    img: &Vol<f32>,
    model: &Path,
    progress: &dyn Fn(usize, usize),
    cancel: &AtomicBool,
    min_size: usize,
) -> Result<Vol<u8>> {
    let session = get_session(model)?;
    let [nx, ny, nz, nt] = img.shape;
    let mut pred = Vol::<u8>::zeros(img.shape);
    let clean = |v: f32| if v.is_nan() { 0.0 } else { v.clamp(f32::MIN, f32::MAX) };
    let volume_wise = model_mode(model) == "xyz";
    let jobs: Vec<(Option<usize>, usize)> = if volume_wise {
        (0..nt).map(|t| (None, t)).collect()
    } else {
        (0..nt).flat_map(|t| (0..nz).map(move |z| (Some(z), t))).collect()
    };
    // Python iterates z outer, t inner; the order does not change the result.
    for (k, &(z, t)) in jobs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::cancelled());
        }
        let zs: Vec<usize> = match z {
            Some(z) => vec![z],
            None => (0..nz).collect(),
        };
        let nzs = zs.len();
        // C order (x, y[, z]) from the Fortran-ordered volume
        let mut x = vec![0f32; nx * ny * nzs];
        let mut vmax = f32::MIN;
        for (kk, &zz) in zs.iter().enumerate() {
            let sl = img.slice(zz, t);
            for j in 0..ny {
                for i in 0..nx {
                    let v = clean(sl[i + nx * j]);
                    x[(i * ny + j) * nzs + kk] = v;
                    vmax = vmax.max(v);
                }
            }
        }
        if vmax > 0.0 {
            for v in &mut x {
                *v /= vmax;
            }
            let dims: Vec<usize> = if volume_wise { vec![nx, ny, nz] } else { vec![nx, ny] };
            let mut labels = predict(&mut session.lock().unwrap(), &x, &dims)?;
            keep_largest_component(&mut labels, &dims, min_size);
            for (kk, &zz) in zs.iter().enumerate() {
                let dst = pred.slice_mut(zz, t);
                for j in 0..ny {
                    for i in 0..nx {
                        dst[i + nx * j] = labels[(i * ny + j) * nzs + kk];
                    }
                }
            }
        }
        progress(k + 1, jobs.len());
    }
    Ok(pred)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_component_per_label() {
        // 1 1 0 1
        // 0 0 0 1
        // 2 0 0 1
        let mut m = vec![1, 1, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1];
        keep_largest_component(&mut m, &[3, 4], 1);
        assert_eq!(m, vec![0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1]);
        let mut m = vec![1, 1, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1];
        keep_largest_component(&mut m, &[3, 4], 2);
        assert_eq!(m, vec![0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
        // tie: the component met first wins
        let mut m = vec![1, 0, 1];
        keep_largest_component(&mut m, &[3], 1);
        assert_eq!(m, vec![1, 0, 0]);
    }

    #[test]
    fn names() {
        assert_eq!(model_mode(Path::new("/m/cine4d_v0006_xy_cine.onnx")), "xy");
        assert_eq!(model_mode(Path::new("cine4d_v0001_xyz_a.onnx")), "xyz");
        assert_eq!(model_version("cine4d_v0006_xy_cine.onnx"), "v0006");
        assert_eq!(model_version("tiny.onnx"), "tiny.onnx");
    }
}
