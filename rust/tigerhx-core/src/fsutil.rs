//! File helpers: atomic writes and path normalisation.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::Result;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A temporary name next to path that no other writer uses at the same time
/// (another process may write the same workspace).
pub fn unique_tmp(path: &Path) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".{}.{}.tmp", std::process::id(), n));
    PathBuf::from(s)
}

/// Write through a temporary file and rename, so an interrupted write (or a
/// reader during a rewrite) never sees a truncated file.
pub fn write_atomic(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> Result<()>) -> Result<()> {
    let tmp = unique_tmp(path);
    let result = (|| {
        let mut f = BufWriter::new(File::create(&tmp)?);
        write(&mut f)?;
        let f = f.into_inner().map_err(|e| e.into_error())?;
        f.sync_all().ok();
        drop(f);
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    result
}

/// Like Python's os.path.abspath: absolute and lexically normalised
/// ('.' and '..' removed), without resolving symbolic links.
pub fn abspath(path: &Path) -> PathBuf {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !matches!(out.components().next_back(), Some(Component::RootDir | Component::Prefix(_)) | None) {
                    out.pop();
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Like os.path.relpath for a path below base (falls back to the path).
pub fn relpath(path: &Path, base: &Path) -> PathBuf {
    path.strip_prefix(base).map(Path::to_path_buf).unwrap_or_else(|_| path.to_path_buf())
}

pub fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn abspath_normalises() {
        assert_eq!(abspath(Path::new("/a/b/../c/./d")), PathBuf::from("/a/c/d"));
        assert_eq!(abspath(Path::new("/../a")), PathBuf::from("/a"));
    }
}
