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
use crate::caselist::{self, CASE_COLUMNS};
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

/// A case to segment, or a result of the original GUI opened for viewing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    #[default]
    Nifti,
    Mat,
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
    #[serde(default)]
    pub kind: Kind,
    /// per-case output choices from a CSV (None = the setting)
    #[serde(default)]
    pub mat_in_inputdir: Option<bool>,
    #[serde(default)]
    pub nii_in_inputdir: Option<bool>,
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
    /// one folder for every .mat ('' = <workspace>/output)
    pub output_dir: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model: String::new(),
            save_mat: true,
            mat_in_inputdir: false,
            nii_in_inputdir: false,
            output_dir: String::new(),
        }
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

fn is_ext(path: &Path, ext: &str) -> bool {
    path.to_string_lossy().to_lowercase().ends_with(ext)
}

/// Original-GUI results in a folder (not its subfolders): *pred*.mat.
pub fn scan_results(folder: &Path) -> Vec<PathBuf> {
    let Ok(rd) = fs::read_dir(folder) else { return Vec::new() };
    let mut found: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            n.ends_with(".mat") && n.contains("pred") && p.is_file()
        })
        .collect();
    found.sort_by_key(|p| path_str(p));
    found
}

/// os.path.commonpath of absolute paths.
fn common_path(paths: &[PathBuf]) -> PathBuf {
    let mut it = paths.iter();
    let Some(first) = it.next() else { return PathBuf::new() };
    let mut common: Vec<_> = first.components().collect();
    for p in it {
        let n = common.iter().zip(p.components()).take_while(|(a, b)| *a == b).count();
        common.truncate(n);
    }
    common.iter().collect()
}

/// "folder/sub/stem" from a path relative to the folder it was added from.
fn case_name(rel: &Path) -> String {
    let head = rel.parent().map(path_str).unwrap_or_default().replace(std::path::MAIN_SEPARATOR, "/");
    let tail = strip_nii_ext(&path_str(rel));
    [head, tail].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("/")
}

/// What adding paths did.
#[derive(Debug, Default)]
pub struct Added {
    pub added: Vec<Case>,
    pub updated: Vec<Case>,
    /// files named in a CSV that do not exist
    pub missing: Vec<String>,
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

    /// The one folder that receives every .mat, as in the original GUI.
    pub fn output_dir(&self) -> PathBuf {
        let s = self.lock().settings.output_dir.trim().to_string();
        if s.is_empty() { self.workspace.join("output") } else { abspath(Path::new(&s)) }
    }

    pub fn csv_dir(&self) -> PathBuf {
        self.workspace.join("csv")
    }

    /// Add NIfTI files and (recursively) folders, case-list CSVs and
    /// original-GUI results (.mat; with `results` a folder adds its
    /// *pred*.mat).
    pub fn add_paths(&self, paths: &[String], include: &str, exclude: &str, results: bool) -> Result<Added> {
        // files are read and folders scanned before taking the lock: a large
        // folder would otherwise hold up every other request meanwhile
        enum Item {
            Csv(Vec<caselist::CaseRow>),
            Mat(Vec<PathBuf>),
            Nifti(Vec<(PathBuf, PathBuf)>),
        }
        let mut plan = Vec::new();
        for p in paths {
            let p = p.trim().trim_matches('"');
            if p.is_empty() {
                continue;
            }
            let p = Path::new(p);
            if p.is_file() && is_ext(p, ".csv") {
                plan.push(Item::Csv(caselist::read_case_csv(p)?));
            } else if (p.is_file() && is_ext(p, ".mat")) || (results && p.is_dir()) {
                plan.push(Item::Mat(if p.is_file() { vec![p.to_path_buf()] } else { scan_results(p) }));
            } else if p.is_dir() {
                plan.push(Item::Nifti(
                    scan_folder(p, include, exclude).into_iter().map(|f| (f.clone(), relpath(&f, p))).collect(),
                ));
            } else if p.is_file() && is_nifti(p) {
                plan.push(Item::Nifti(vec![(p.to_path_buf(), PathBuf::from(p.file_name().unwrap_or_default()))]));
            }
        }
        let mut out = Added::default();
        {
            let mut state = self.lock();
            let push = |state: &mut State, path: &Path, name: String, kind: Kind| -> Option<Case> {
                let id = case_id(path);
                if state.case(&id).is_some() {
                    return None;
                }
                let mut c = Case::new(path, name);
                c.kind = kind;
                state.cases.push(c.clone());
                Some(c)
            };
            for item in plan {
                match item {
                    Item::Csv(rows) => Self::import_csv(&mut state, rows, &mut out),
                    Item::Mat(files) => {
                        for f in files {
                            let name = path_str(Path::new(f.file_name().unwrap_or_default()));
                            let name = name[..name.len() - 4].to_string();
                            out.added.extend(push(&mut state, &f, name, Kind::Mat));
                        }
                    }
                    Item::Nifti(items) => {
                        for (path, rel) in items {
                            out.added.extend(push(&mut state, &path, case_name(&rel), Kind::Nifti));
                        }
                    }
                }
            }
        }
        self.save()?;
        Ok(out)
    }

    /// An original-GUI case list: a filled-in Apex counts as confirmed; the
    /// two output columns override the settings for that case.  Names are
    /// relative to the folder the files have in common.
    fn import_csv(state: &mut State, rows: Vec<caselist::CaseRow>, out: &mut Added) {
        out.missing.extend(rows.iter().filter(|r| !r.path.is_file()).map(|r| path_str(&r.path)));
        let rows: Vec<_> = rows.into_iter().filter(|r| r.path.is_file()).collect();
        let dirs: Vec<PathBuf> = rows.iter().map(|r| r.path.parent().map(Path::to_path_buf).unwrap_or_default()).collect();
        let common = common_path(&dirs);
        for r in rows {
            let id = case_id(&r.path);
            let new = state.case(&id).is_none();
            if new {
                state.cases.push(Case::new(&r.path, case_name(&relpath(&r.path, &common))));
            }
            let c = state.case_mut(&id).unwrap();
            if !new && c.status.busy() {
                continue;
            }
            c.mat_in_inputdir = r.mat_in_inputdir;
            c.nii_in_inputdir = r.nii_in_inputdir;
            if let Some(a) = r.apex {
                c.apex_slice = Some(a);
                c.apex_confirmed = true;
            }
            if new { out.added.push(c.clone()) } else { out.updated.push(c.clone()) }
        }
    }

    /// The original GUI's GenCSV: one row per NIfTI found, written to
    /// <workspace>/csv/files_<time>.csv for filling in the Apex column.
    /// Returns the file and the number of rows.
    pub fn gen_csv(&self, folders: &[String], include: &str, exclude: &str) -> Result<(PathBuf, usize)> {
        let mut files = Vec::new();
        for f in folders {
            let f = Path::new(f.trim().trim_matches('"'));
            if f.is_dir() {
                files.extend(scan_folder(f, include, exclude));
            }
        }
        let s = self.settings();
        fs::create_dir_all(self.csv_dir())?;
        let stamp = chrono::Local::now().format("%y%m%d_%H%M%S");
        let path = self.csv_dir().join(format!("files_{stamp}.csv"));
        let rows: Vec<Vec<String>> =
            files.iter().map(|f| caselist::gen_row(f, s.mat_in_inputdir, s.nii_in_inputdir)).collect();
        caselist::write_csv(&path, &CASE_COLUMNS, &rows)?;
        Ok((path, files.len()))
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
        assert!(s.ends_with(r#""levels":null,"crop":null,"outputs":{},"kind":"nifti","mat_in_inputdir":null,"nii_in_inputdir":null}"#));
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
