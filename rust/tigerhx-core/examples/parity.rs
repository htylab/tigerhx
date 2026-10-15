//! Drives the pipeline from the command line, for the comparison with the
//! Python core in tests/test_rust_parity.py.
//!
//!   parity run <workspace> <model> <nifti|csv|mat>...   add; segment NIfTI, open .mat
//!   parity results <workspace> <folder>          open the folder's *pred*.mat
//!   parity apex <workspace> <id> <apex> <base_first: 1|0|none>
//!   parity display <workspace> <id> <image|seg|aha|sectors> <crop: 0|1> <out.nii>
//!   parity settings <workspace> <save_mat> <mat_in_inputdir> <nii_in_inputdir>

use std::path::Path;
use std::sync::atomic::AtomicBool;

use tigerhx_core::pipeline::{self, VolumeKind};
use tigerhx_core::project::{Case, Kind, Project, Status};

fn process(project: &Project, model: Option<&Path>, cases: Vec<Case>) -> Result<(), Box<dyn std::error::Error>> {
    let cancel = AtomicBool::new(false);
    for c in cases {
        if c.kind == Kind::Mat {
            pipeline::import_mat_case(project, &c.id)?;
        } else if let Some(model) = model {
            pipeline::segment_case(project, &c.id, model, &|_, _| {}, &cancel)?;
        }
        project.update(&c.id, |c| c.status = if c.apex_confirmed { Status::Done } else { Status::Review });
    }
    project.save()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let project = Project::open(Path::new(&args[1]))?;
    match args[0].as_str() {
        "settings" => {
            {
                let mut st = project.lock();
                st.settings.save_mat = args[2] == "1";
                st.settings.mat_in_inputdir = args[3] == "1";
                st.settings.nii_in_inputdir = args[4] == "1";
            }
            project.save()?;
        }
        "run" => {
            let added = project.add_paths(&args[3..], "", "", false)?;
            process(&project, Some(Path::new(&args[2])), added.added)?;
        }
        "results" => {
            let added = project.add_paths(&args[2..], "", "", true)?;
            process(&project, None, added.added)?;
        }
        "apex" => {
            let base_first = match args[4].as_str() {
                "1" => Some(true),
                "0" => Some(false),
                _ => None,
            };
            pipeline::set_apex(&project, &args[2], args[3].parse()?, base_first)?;
            pipeline::write_outputs(&project, &args[2], None, None)?;
            project.save()?;
        }
        "display" => {
            let case = project.case(&args[2]).ok_or("no such case")?;
            let kind = VolumeKind::parse(&args[3]).ok_or("bad kind")?;
            let bytes = pipeline::display_volume(&project, &case, kind, args[4] == "1")?;
            std::fs::write(&args[5], bytes)?;
        }
        other => return Err(format!("unknown command {other}").into()),
    }
    Ok(())
}
