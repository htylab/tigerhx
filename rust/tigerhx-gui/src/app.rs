//! Project state plus a single background worker (the Rust version of
//! tigerhx/webgui/server.py's App).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use serde::Serialize;
use serde_json::{json, Value};
use tigerhx_core::models::{self, GUI_MODELS};
use tigerhx_core::pipeline::{self, VolumeKind};
use tigerhx_core::project::{Kind, Project, Status};
use tigerhx_core::{Error, Result};

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    pub state: String,
    pub name: String,
    pub progress: f64,
    pub message: String,
}

type CacheKey = (String, VolumeKind, bool);

/// A message for the GUI's log panel (the original GUI's log box).
#[derive(Clone, Debug, Serialize)]
pub struct LogEntry {
    pub id: u64,
    pub time: String,
    pub level: &'static str,
    pub text: String,
}

pub struct App {
    pub project: Project,
    pub token: String,
    jobs: Mutex<VecDeque<(String, String)>>,
    jobs_ready: Condvar,
    cancel: AtomicBool,
    current: Mutex<Option<String>>,
    model_status: Mutex<ModelStatus>,
    cache: Mutex<VecDeque<(CacheKey, Arc<Vec<u8>>)>>,
    /// one reanalysis at a time
    analysis: Mutex<()>,
    log: Mutex<(u64, VecDeque<LogEntry>)>,
}

fn new_token() -> String {
    let mut b = [0u8; 24];
    getrandom::fill(&mut b).expect("no random source");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn log_error(what: &str, e: &Error) {
    log!("TigerHx: {what}: {e}");
}

impl App {
    pub fn new(workspace: &Path) -> Result<Arc<App>> {
        let app = Arc::new(App {
            project: Project::open(workspace)?,
            token: new_token(),
            jobs: Mutex::new(VecDeque::new()),
            jobs_ready: Condvar::new(),
            cancel: AtomicBool::new(false),
            current: Mutex::new(None),
            model_status: Mutex::new(ModelStatus {
                state: "idle".into(),
                name: String::new(),
                progress: 0.0,
                message: String::new(),
            }),
            cache: Mutex::new(VecDeque::new()),
            analysis: Mutex::new(()),
            log: Mutex::new((0, VecDeque::new())),
        });
        app.log(format!("Workspace: {}", app.project.workspace.display()), "info");
        let worker = app.clone();
        std::thread::spawn(move || worker.worker());
        let refresher = app.clone();
        std::thread::spawn(move || refresher.refresh_old_cases());
        Ok(app)
    }

    fn save(&self) {
        if let Err(e) = self.project.save() {
            log_error("saving project.json", &e);
        }
    }

    // ---- log ---------------------------------------------------------
    pub fn log(&self, text: impl Into<String>, level: &'static str) {
        let mut log = self.log.lock().unwrap();
        log.0 += 1;
        let entry =
            LogEntry { id: log.0, time: chrono::Local::now().format("%H:%M:%S").to_string(), level, text: text.into() };
        log.1.push_back(entry);
        while log.1.len() > 2000 {
            log.1.pop_front();
        }
    }

    pub fn log_since(&self, after: u64) -> Vec<LogEntry> {
        self.log.lock().unwrap().1.iter().filter(|e| e.id > after).cloned().collect()
    }

    // ---- state -------------------------------------------------------
    pub fn state(&self) -> Value {
        let (cases, mut settings) = {
            let st = self.project.lock();
            (serde_json::to_value(&st.cases).unwrap(), st.settings.clone())
        };
        let available = models::list_models(None);
        let mut choices: Vec<String> = available.clone();
        for m in GUI_MODELS {
            if !choices.iter().any(|c| c == m) {
                choices.push(m.to_string());
            }
        }
        choices.sort();
        if settings.model.is_empty() || !choices.contains(&settings.model) {
            settings.model = available.last().cloned().unwrap_or_else(|| GUI_MODELS[GUI_MODELS.len() - 1].into());
        }
        let models: Vec<Value> =
            choices.iter().map(|m| json!({"name": m, "downloaded": available.contains(m)})).collect();
        json!({
            "cases": cases,
            "settings": settings,
            "workspace": self.project.workspace.to_string_lossy(),
            "output_dir": self.project.output_dir().to_string_lossy(),
            "models": models,
            "running": *self.current.lock().unwrap(),
            "queued": self.jobs.lock().unwrap().len(),
            "model_status": *self.model_status.lock().unwrap(),
        })
    }

    /// Cases analysed by an older version lack the zoom box and have the
    /// old checks: redo their (cheap) analysis once.
    fn refresh_old_cases(&self) {
        let old: Vec<String> = {
            let st = self.project.lock();
            st.cases
                .iter()
                .filter(|c| {
                    c.status.segmented()
                        && (c.crop.is_none() || !pipeline::sectors_path(&self.project, c).is_file())
                })
                .map(|c| c.id.clone())
                .collect()
        };
        for id in &old {
            let Some(case) = self.project.case(id) else { continue };
            if case.status.segmented() && pipeline::has_seg(&self.project, &case) {
                let _guard = self.analysis.lock().unwrap();
                match pipeline::analyze_case(&self.project, id, None) {
                    Ok(_) => self.invalidate(id, None),
                    Err(e) => log_error(&format!("analysing {}", case.name), &e),
                }
            }
        }
        if !old.is_empty() {
            self.save();
        }
    }

    pub fn update_settings(&self, values: &serde_json::Map<String, Value>) -> Result<()> {
        {
            let mut st = self.project.lock();
            let s = &mut st.settings;
            if let Some(v) = values.get("model") {
                s.model = v.as_str().ok_or_else(|| Error::invalid("model must be a string"))?.trim().to_string();
            }
            if let Some(v) = values.get("output_dir") {
                s.output_dir = match v {
                    Value::String(t) => t.trim().to_string(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
            }
            for (k, field) in [
                ("save_mat", &mut s.save_mat),
                ("mat_in_inputdir", &mut s.mat_in_inputdir),
                ("nii_in_inputdir", &mut s.nii_in_inputdir),
            ] {
                if let Some(v) = values.get(k) {
                    *field = truthy(v);
                }
            }
        }
        self.project.save()
    }

    // ---- cases -------------------------------------------------------
    /// Add NIfTI files/folders, case-list CSVs or original-GUI results;
    /// results are opened right away by the worker.
    pub fn add(&self, paths: &[String], include: &str, exclude: &str, results: bool) -> Result<Value> {
        let r = self.project.add_paths(paths, include, exclude, results)?;
        for p in &r.missing {
            self.log(format!("Not found, skipped: {p}"), "warn");
        }
        let mats: Vec<String> = r.added.iter().filter(|c| c.kind == Kind::Mat).map(|c| c.id.clone()).collect();
        let csv_apex = r.added.iter().chain(&r.updated).filter(|c| c.kind != Kind::Mat && c.apex_confirmed).count();
        let mut msg = format!("Added {} case(s)", r.added.len());
        if !r.updated.is_empty() {
            msg += &format!(", updated {}", r.updated.len());
        }
        if csv_apex > 0 {
            msg += &format!(", {csv_apex} with the apex from the CSV");
        }
        self.log(msg, "info");
        if !mats.is_empty() {
            self.queue(&mats, "");
        }
        Ok(json!({"added": r.added.len(), "updated": r.updated.len(), "missing": r.missing}))
    }

    pub fn gen_csv(&self, folders: &[String], include: &str, exclude: &str) -> Result<Value> {
        let (path, n) = self.project.gen_csv(folders, include, exclude)?;
        self.log(
            format!("GenCSV: {n} file(s) listed in {}; fill in the Apex column and add the CSV", path.display()),
            "info",
        );
        Ok(json!({"path": path.to_string_lossy(), "count": n}))
    }

    /// Fetch every GUI model now (for a computer that is usually offline).
    pub fn download_all(self: &Arc<Self>) -> usize {
        let dir = models::model_dir();
        let missing: Vec<String> =
            GUI_MODELS.iter().filter(|m| !dir.join(m).is_file()).map(|m| m.to_string()).collect();
        let n = missing.len();
        if n == 0 {
            self.log("All models are already downloaded", "info");
            return 0;
        }
        let app = self.clone();
        std::thread::spawn(move || {
            for name in missing {
                app.log(format!("Downloading {name} ..."), "info");
                *app.model_status.lock().unwrap() =
                    ModelStatus { state: "downloading".into(), name: name.clone(), progress: 0.0, message: String::new() };
                let progress = |done: u64, total: u64| {
                    app.model_status.lock().unwrap().progress = if total > 0 { done as f64 / total as f64 } else { 0.0 };
                };
                match models::ensure_model(&name, &progress, None) {
                    Ok(_) => app.log(format!("Downloaded {name}"), "info"),
                    Err(e) => {
                        app.log(format!("Download failed: {name}: {e}"), "error");
                        let mut ms = app.model_status.lock().unwrap();
                        ms.state = "error".into();
                        ms.message = e.to_string();
                        return;
                    }
                }
            }
            app.model_status.lock().unwrap().state = "idle".into();
        });
        n
    }

    // ---- jobs --------------------------------------------------------
    pub fn run(&self, ids: &[String]) -> usize {
        let (model, targets) = {
            let st = self.project.lock();
            let model = if st.settings.model.is_empty() {
                GUI_MODELS[GUI_MODELS.len() - 1].to_string()
            } else {
                st.settings.model.clone()
            };
            let targets: Vec<String> = if ids.is_empty() {
                st.cases
                    .iter()
                    .filter(|c| matches!(c.status, Status::Pending | Status::Error) && c.kind != Kind::Mat)
                    .map(|c| c.id.clone())
                    .collect()
            } else {
                ids.to_vec()
            };
            (model, targets)
        };
        self.queue(&targets, &model)
    }

    /// Queue cases for the worker; an empty model opens results.
    fn queue(&self, ids: &[String], model: &str) -> usize {
        let mut queued = Vec::new();
        {
            let mut st = self.project.lock();
            for id in ids {
                let Some(c) = st.case_mut(id) else { continue };
                if c.status.busy() {
                    continue;
                }
                if c.kind == Kind::Mat && c.status.segmented() {
                    continue; // an opened result is read once; there is nothing to re-run
                }
                c.status = Status::Queued;
                c.progress = 0.0;
                c.message.clear();
                queued.push(id.clone());
            }
        }
        self.save();
        self.cancel.store(false, Ordering::SeqCst);
        let n = queued.len();
        {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.extend(queued.into_iter().map(|id| (id, model.to_string())));
            self.jobs_ready.notify_all();
        }
        if n > 0 && !model.is_empty() {
            self.log(format!("Queued {n} case(s) with {model}"), "info");
        }
        n
    }

    pub fn stop(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        let drained: Vec<(String, String)> = self.jobs.lock().unwrap().drain(..).collect();
        for (id, _) in drained {
            self.project.update(&id, |c| {
                if c.status == Status::Queued {
                    c.status = Status::Pending;
                }
            });
        }
        self.save();
    }

    fn next_job(&self) -> (String, String) {
        let mut jobs = self.jobs.lock().unwrap();
        loop {
            if let Some(j) = jobs.pop_front() {
                return j;
            }
            jobs = self.jobs_ready.wait(jobs).unwrap();
        }
    }

    fn worker(&self) {
        loop {
            let (id, model) = self.next_job();
            match self.project.case(&id) {
                Some(c) if c.status == Status::Queued => {}
                _ => continue,
            }
            *self.current.lock().unwrap() = Some(id.clone());
            let name = self.project.case(&id).map(|c| c.name).unwrap_or_default();
            let is_mat = self.project.case(&id).is_some_and(|c| c.kind == Kind::Mat);
            let result = if is_mat { self.open_result(&id) } else { self.segment(&id, &model) };
            match result {
                Ok(()) => {}
                Err(e) if e.is_cancelled() => {
                    self.project.update(&id, |c| {
                        c.status = Status::Pending;
                        c.progress = 0.0;
                        c.message = "stopped".into();
                    });
                    self.log(format!("Stopped {name}"), "warn");
                }
                Err(e) => {
                    log_error(&format!("segmenting {id}"), &e);
                    self.project.update(&id, |c| {
                        c.status = Status::Error;
                        c.message = e.to_string();
                    });
                    self.log(format!("{name}: {e}"), "error");
                    let mut ms = self.model_status.lock().unwrap();
                    if ms.state == "downloading" || e.message.contains("download") {
                        ms.state = "error".into();
                        ms.message = e.to_string();
                    }
                }
            }
            *self.current.lock().unwrap() = None;
            self.invalidate(&id, None);
            self.save();
        }
    }

    fn segment(&self, id: &str, model: &str) -> Result<()> {
        if !models::model_dir().join(model).is_file() {
            *self.model_status.lock().unwrap() =
                ModelStatus { state: "downloading".into(), name: model.into(), progress: 0.0, message: String::new() };
        }
        let progress = |done: u64, total: u64| {
            self.model_status.lock().unwrap().progress = if total > 0 { done as f64 / total as f64 } else { 0.0 };
        };
        let path = models::ensure_model(model, &progress, None);
        {
            let mut ms = self.model_status.lock().unwrap();
            if ms.state == "downloading" {
                ms.state = "idle".into();
            }
        }
        let path = path?;
        self.project.update(id, |c| {
            c.status = Status::Running;
            c.progress = 0.0;
        });
        let name = self.project.case(id).map(|c| c.name).unwrap_or_default();
        self.log(format!("Segmenting {name} ..."), "info");
        let progress = |done: usize, total: usize| {
            self.project.update(id, |c| c.progress = done as f64 / total as f64);
        };
        pipeline::segment_case(&self.project, id, &path, &progress, &self.cancel)?;
        self.project.update(id, |c| {
            c.status = if c.apex_confirmed { Status::Done } else { Status::Review };
            c.message.clear();
        });
        if let Some(c) = self.project.case(id) {
            let mut msg = format!(
                "Finished {}: apex {}{}",
                c.name,
                c.apex_slice.map(|a| a.to_string()).unwrap_or_else(|| "None".into()),
                if c.apex_confirmed { " (confirmed)" } else { " (suggested)" }
            );
            if let Some(ch) = c.checks.as_ref().filter(|ch| !ch.is_empty()) {
                msg += &format!(", {} slice(s) to check", ch.len());
            }
            for v in c.outputs.values() {
                msg += &format!("; wrote {}", v.as_str().unwrap_or_default());
            }
            self.log(msg, "info");
        }
        Ok(())
    }

    /// Open an original-GUI result (read-only case).
    fn open_result(&self, id: &str) -> Result<()> {
        self.project.update(id, |c| {
            c.status = Status::Running;
            c.progress = 0.0;
        });
        pipeline::import_mat_case(&self.project, id)?;
        self.project.update(id, |c| {
            c.status = if c.apex_confirmed { Status::Done } else { Status::Review };
            c.message.clear();
        });
        if let Some(c) = self.project.case(id) {
            let apex = if c.apex_confirmed { format!(" (apex {})", c.apex_slice.unwrap_or(0)) } else { String::new() };
            self.log(format!("Opened {}{apex}", c.name), "info");
        }
        Ok(())
    }

    // ---- analysis ----------------------------------------------------
    /// The user's apex choice for a segmented case: reanalyse, refresh the
    /// AHA volume and project file and (off-thread) rewrite the outputs.
    pub fn set_apex(self: &Arc<Self>, id: &str, apex: i64, base_first: Option<bool>) -> Result<()> {
        let case = self.project.case(id).ok_or_else(|| Error::not_found("not found"))?;
        if let Some(shape) = &case.shape {
            let nz = shape[2] as i64;
            if !(0..nz).contains(&apex) {
                return Err(Error::invalid(format!("apex slice must be within 0..{}", nz - 1)));
            }
        } else if apex < 0 {
            return Err(Error::invalid("apex slice must not be negative"));
        }
        if case.status.busy() || !pipeline::has_seg(&self.project, &case) {
            return Err(Error::invalid("case is not segmented"));
        }
        {
            let _guard = self.analysis.lock().unwrap();
            pipeline::set_apex(&self.project, id, apex as usize, base_first)?;
        }
        self.invalidate(id, Some(VolumeKind::Aha));
        self.invalidate(id, Some(VolumeKind::Sectors));
        if let Some(c) = self.project.case(id) {
            let order = if c.base_first == Some(false) { " (apex -> base order)" } else { "" };
            self.log(format!("{}: apex set to slice {}{order}", c.name, c.apex_slice.unwrap_or(0)), "info");
        }
        self.save();
        let app = self.clone();
        let id = id.to_string();
        std::thread::spawn(move || app.write_outputs(&id));
        Ok(())
    }

    fn write_outputs(&self, id: &str) {
        match pipeline::write_outputs(&self.project, id, None, None) {
            Ok(()) => self.save(),
            Err(e) => {
                log_error("writing outputs", &e);
                self.project.update(id, |c| c.message = format!("output: {e}"));
                let name = self.project.case(id).map(|c| c.name).unwrap_or_default();
                self.log(format!("{name}: output: {e}"), "error");
            }
        }
    }

    // ---- volumes for the viewer --------------------------------------
    pub fn invalidate(&self, id: &str, kind: Option<VolumeKind>) {
        self.cache.lock().unwrap().retain(|((cid, k, _), _)| !(cid == id && kind.is_none_or(|kind| *k == kind)));
    }

    /// Display NIfTI bytes (cached, the 8 most recent).
    pub fn volume(&self, id: &str, kind: VolumeKind, crop: bool) -> Result<Arc<Vec<u8>>> {
        let key = (id.to_string(), kind, crop);
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache.remove(pos).unwrap();
                let body = entry.1.clone();
                cache.push_back(entry);
                return Ok(body);
            }
        }
        let case = self.project.case(id).ok_or_else(|| Error::not_found("not found"))?;
        let body = Arc::new(pipeline::display_volume(&self.project, &case, kind, crop)?);
        let mut cache = self.cache.lock().unwrap();
        cache.push_back((key, body.clone()));
        while cache.len() > 8 {
            cache.pop_front();
        }
        Ok(body)
    }

    /// The case folder, the output or CSV folder, or the workspace, to show
    /// in the file manager.
    pub fn folder(&self, id: Option<&str>, folder: Option<&str>) -> PathBuf {
        if let Some(c) = id.and_then(|id| self.project.case(id)) {
            return self.project.case_dir(&c);
        }
        let dir = match folder {
            Some("output") => self.project.output_dir(),
            Some("csv") => self.project.csv_dir(),
            _ => return self.project.workspace.clone(),
        };
        std::fs::create_dir_all(&dir).ok();
        dir
    }
}

/// Python's bool() of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
