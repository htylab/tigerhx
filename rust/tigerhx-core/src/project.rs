//! A workspace holding a list of cases and their analysis state.
//!
//! project.json has the same format as the Python GUI's, so either program
//! can open a workspace written by the other.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use crate::aha::Level;
use crate::error::Result;
use crate::fsutil::{abspath, path_str, relpath, write_atomic};
use crate::legacy::{safe_name, strip_nii_ext};
use crate::metrics::Check;

/// pending -> queued -> running -> review (segmented, apex not confirmed) -> done
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Pending,
    Queued,
    Running,
    Review,
    Done,
    Error,
}

impl Status {
    pub fn busy(self) -> bool {
        matches!(self, Status::Queued | Status::Running)
    }
    pub fn segmented(self) -> bool {
        matches!(self, Status::Review | Status::Done)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub path: String,
    pub name: String,
    #[serde(default)]
    pub status: Status,
    #[serde(default)]
    pub progress: f64,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub shape: Option<Vec<usize>>,
    #[serde(default)]
    pub zooms: Option<Vec<f64>>,
    #[serde(default)]
    pub ed: Option<usize>,
    #[serde(default)]
    pub es: Option<usize>,
    #[serde(default)]
    pub base_first: Option<bool>,
    #[serde(default)]
    pub apex_slice: Option<usize>,
    #[serde(default)]
    pub apex_confirmed: bool,
    /// slices whose segmentation is worth a look (metrics::slice_checks)
    #[serde(default)]
    pub checks: Option<Vec<Check>>,
    #[serde(default)]
    pub levels: Option<Vec<Option<Level>>>,
    /// [x0, x1, y0, y1] around the heart, for the zoomed view
    #[serde(default)]
    pub crop: Option<[usize; 4]>,
    #[serde(default)]
    pub outputs: serde_json::Map<String, serde_json::Value>,
}

impl Case {
    pub fn new(path: &Path, name: String) -> Self {
        Case { id: case_id(path), path: path_str(&abspath(path)), name, ..Default::default() }
    }

    /// Folder name below <workspace>/cases.
    pub fn folder(&self) -> String {
        let name: String = safe_name(&self.name).chars().take(60).collect();
        format!("{name}_{}", &self.id[..self.id.len().min(6)])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub model: String,
    /// legacy .mat in the workspace
    pub save_mat: bool,
    /// legacy .mat next to the input file
    pub mat_in_inputdir: bool,
    /// label NIfTI next to the input file
    pub nii_in_inputdir: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { model: String::new(), save_mat: true, mat_in_inputdir: false, nii_in_inputdir: false }
    }
}

impl Settings {
    pub fn writes_outputs(&self) -> bool {
        self.save_mat || self.mat_in_inputdir || self.nii_in_inputdir
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub settings: Settings,
    #[serde(default)]
    pub cases: Vec<Case>,
}

impl State {
    pub fn case(&self, id: &str) -> Option<&Case> {
        self.cases.iter().find(|c| c.id == id)
    }
    pub fn case_mut(&mut self, id: &str) -> Option<&mut Case> {
        self.cases.iter_mut().find(|c| c.id == id)
    }
}

pub fn case_id(path: &Path) -> String {
    let digest = Sha1::digest(path_str(&abspath(path)).as_bytes());
    digest.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

fn terms(text: &str) -> Vec<String> {
    text.replace(';', ",").split(',').map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()).collect()
}

pub fn is_nifti(path: &Path) -> bool {
    let n = path.to_string_lossy().to_lowercase();
    n.ends_with(".nii") || n.ends_with(".nii.gz")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            walk(&path, out); // like os.walk, symbolic links to folders are not followed
        } else if ft.is_file() || (ft.is_symlink() && path.is_file()) {
            out.push(path);
        }
    }
}

/// NIfTI files below folder whose path relative to folder contains any
/// include term (all files if none) and no exclude term (case-insensitive).
pub fn scan_folder(folder: &Path, include: &str, exclude: &str) -> Vec<PathBuf> {
    let (inc, exc) = (terms(include), terms(exclude));
    let mut files = Vec::new();
    walk(folder, &mut files);
    let mut found: Vec<PathBuf> = files
        .into_iter()
        .filter(|p| is_nifti(p))
        .filter(|p| {
            let rel = path_str(&relpath(p, folder)).to_lowercase();
            (inc.is_empty() || inc.iter().any(|t| rel.contains(t.as_str())))
                && !exc.iter().any(|t| rel.contains(t.as_str()))
        })
        .collect();
    found.sort_by_key(|p| path_str(p));
    found
}

pub struct Project {
    pub workspace: PathBuf,
    file: PathBuf,
    state: Mutex<State>,
}

impl Project {
    pub fn open(workspace: &Path) -> Result<Project> {
        let workspace = abspath(workspace);
        fs::create_dir_all(workspace.join("cases"))?;
        let file = workspace.join("project.json");
        let mut state = State::default();
        if file.is_file() {
            state = serde_json::from_slice(&fs::read(&file)?)?;
            for c in &mut state.cases {
                if c.status.busy() {
                    // interrupted by a restart
                    c.status = Status::Pending;
                    c.progress = 0.0;
                }
            }
        }
        Ok(Project { workspace, file, state: Mutex::new(state) })
    }

    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A copy of one case.
    pub fn case(&self, id: &str) -> Option<Case> {
        self.lock().case(id).cloned()
    }

    /// Change one case under the lock; false if it no longer exists.
    pub fn update(&self, id: &str, f: impl FnOnce(&mut Case)) -> bool {
        match self.lock().case_mut(id) {
            Some(c) => {
                f(c);
                true
            }
            None => false,
        }
    }

    pub fn settings(&self) -> Settings {
        self.lock().settings.clone()
    }

    pub fn save(&self) -> Result<()> {
        let state = self.lock();
        let mut out = Vec::new();
        let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
        let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
        state.serialize(&mut ser)?;
        // another process may save the same workspace: unique temporary file
        write_atomic(&self.file, |f| {
            use std::io::Write;
            f.write_all(&out)?;
            Ok(())
        })
    }

    pub fn case_dir(&self, case: &Case) -> PathBuf {
        self.workspace.join("cases").join(case.folder())
    }

    /// Add files and (recursively) folders; returns the new cases.
    pub fn add_paths(&self, paths: &[String], include: &str, exclude: &str) -> Result<Vec<Case>> {
        let mut added = Vec::new();
        {
            let mut state = self.lock();
            for p in paths {
                let p = p.trim().trim_matches('"');
                if p.is_empty() {
                    continue;
                }
                let p = Path::new(p);
                let items: Vec<(PathBuf, PathBuf)> = if p.is_dir() {
                    scan_folder(p, include, exclude).into_iter().map(|f| (f.clone(), relpath(&f, p))).collect()
                } else if p.is_file() && is_nifti(p) {
                    vec![(p.to_path_buf(), PathBuf::from(p.file_name().unwrap_or_default()))]
                } else {
                    continue;
                };
                for (path, rel) in items {
                    let id = case_id(&path);
                    if state.case(&id).is_some() {
                        continue;
                    }
                    let head = rel.parent().map(path_str).unwrap_or_default().replace(std::path::MAIN_SEPARATOR, "/");
                    let tail = strip_nii_ext(&path_str(&rel));
                    let name = [head, tail].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("/");
                    let c = Case::new(&path, name);
                    state.cases.push(c.clone());
                    added.push(c);
                }
            }
        }
        self.save()?;
        Ok(added)
    }

    /// Remove cases from the list (not queued or running ones); files are kept.
    pub fn remove(&self, ids: &[String]) -> Result<()> {
        self.lock().cases.retain(|c| !ids.contains(&c.id) || c.status.busy());
        self.save()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_json_matches_python_layout() {
        let c = Case { id: "abc123def456".into(), path: "/x/a.nii".into(), name: "a".into(), ..Default::default() };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.starts_with(r#"{"id":"abc123def456","path":"/x/a.nii","name":"a","status":"pending","progress":0.0,"message":"","model":"","shape":null"#));
        assert!(s.ends_with(r#""levels":null,"crop":null,"outputs":{}}"#));
        let back: Case = serde_json::from_str(r#"{"id":"x","path":"p","name":"n","status":"review","unknown":1}"#).unwrap();
        assert_eq!(back.status, Status::Review);
        assert_eq!(c.folder(), "a_abc123");
    }

    #[cfg(unix)]
    #[test]
    fn case_id_matches_python() {
        // hashlib.sha1(b'/data/cine.nii.gz').hexdigest()[:12]
        assert_eq!(case_id(Path::new("/data/cine.nii.gz")), "bbf496933514");
    }
}
