//! Model files: location, listing and download.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

pub const MODEL_SERVERS: [&str; 4] = [
    "https://github.com/htylab/tigerhx/releases/download/modelhub/",
    "https://github.com/htylab/tigerbx/releases/download/modelhub/",
    "https://data.mrilab.org/onnxmodel/dev/",
    "https://data.mrilab.org/onnxmodel/releasev1/",
];

pub const GUI_MODELS: [&str; 3] =
    ["cine4d_v0004_xy_retrain.onnx", "cine4d_v0005_xy_cineT1map.onnx", "cine4d_v0006_xy_cine.onnx"];

/// Folder of the executable.
pub fn app_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// $TIGERHX_MODEL_DIR or `models` next to the executable (created if needed).
pub fn model_dir() -> PathBuf {
    let dir = std::env::var_os("TIGERHX_MODEL_DIR")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| app_dir().join("models"));
    fs::create_dir_all(&dir).ok();
    dir
}

/// Sorted .onnx file names in the model folder.
pub fn list_models(dir: Option<&Path>) -> Vec<String> {
    let dir = dir.map(Path::to_path_buf).unwrap_or_else(model_dir);
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".onnx"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn agent() -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    ureq::Agent::config_builder()
        .tls_config(tls)
        .proxy(ureq::Proxy::try_from_env())
        .build()
        .into()
}

/// Download to `<file>.part` and rename, so an interrupted download never
/// leaves a truncated model behind.  progress(bytes_done, bytes_total).
pub fn download(url: &str, file: &Path, progress: &dyn Fn(u64, u64)) -> Result<()> {
    let part = PathBuf::from(format!("{}.part", file.display()));
    let result = (|| -> Result<()> {
        let response = agent().get(url).call().map_err(|e| Error::other(e.to_string()))?;
        let total: u64 = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut reader = response.into_body().into_reader();
        let mut out = fs::File::create(&part)?;
        let mut buf = vec![0u8; 1 << 20];
        let mut done = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            progress(done, total);
        }
        out.sync_all().ok();
        drop(out);
        fs::rename(&part, file)?;
        Ok(())
    })();
    if part.exists() {
        fs::remove_file(&part).ok();
    }
    result
}

/// Path of a model, downloading it from the model servers if needed.
pub fn ensure_model(name: &str, progress: &dyn Fn(u64, u64), dir: Option<&Path>) -> Result<PathBuf> {
    if Path::new(name).is_file() {
        return Ok(PathBuf::from(name));
    }
    let mut name = name.to_string();
    if !name.ends_with(".onnx") {
        name.push_str(".onnx");
    }
    let base = Path::new(&name).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(name.clone());
    let dir = dir.map(Path::to_path_buf).unwrap_or_else(model_dir);
    let path = dir.join(&base);
    if path.is_file() {
        return Ok(path);
    }
    let mut errors = Vec::new();
    for server in MODEL_SERVERS {
        match download(&format!("{server}{base}"), &path, progress) {
            Ok(()) => return Ok(path),
            Err(e) => errors.push(format!("{server}: {e}")),
        }
    }
    Err(Error::other(format!(
        "Cannot download {name}. Check the model name or internet connection.\n{}",
        errors.join("\n")
    )))
}
