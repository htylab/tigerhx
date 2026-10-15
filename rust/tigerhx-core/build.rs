//! Builds a model into the library when one is at hand, as rustbx does: the
//! default GUI model in rust/models/ (CI downloads it there) or the file
//! named by TIGERHX_EMBED_MODEL.  The executable then segments without a
//! download; other models are still downloaded on first use.

use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(embedded_model)");
    println!("cargo:rerun-if-env-changed=TIGERHX_EMBED_MODEL");
    println!("cargo:rerun-if-changed=build.rs");
    let path = match std::env::var_os("TIGERHX_EMBED_MODEL") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../models/cine4d_v0006_xy_cine.onnx"),
    };
    if let Ok(path) = path.canonicalize() {
        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
            println!("cargo:rustc-cfg=embedded_model");
            println!("cargo:rustc-env=TIGERHX_EMBEDDED_MODEL={}", path.display());
            println!("cargo:rustc-env=TIGERHX_EMBEDDED_MODEL_NAME={}", path.file_name().unwrap().to_string_lossy());
        }
    }
}
