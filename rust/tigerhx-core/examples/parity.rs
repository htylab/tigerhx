//! Drives the pipeline from the command line, for the comparison with the
//! Python core in tests/test_rust_parity.py.
//!
//!   parity run <workspace> <model> <nifti>...   add, segment, print project.json
//!   parity apex <workspace> <id> <apex> <base_first: 1|0|none>
//!   parity display <workspace> <id> <image|seg|aha> <crop: 0|1> <out.nii>
//!   parity settings <workspace> <save_mat> <mat_in_inputdir> <nii_in_inputdir>

use std::path::Path;
use std::sync::atomic::AtomicBool;

use tigerhx_core::pipeline::{self, VolumeKind};
use tigerhx_core::project::{Project, Status};

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
            let model = Path::new(&args[2]);
            let added = project.add_paths(&args[3..], "", "")?;
            let cancel = AtomicBool::new(false);
            for c in added {
                pipeline::segment_case(&project, &c.id, model, &|_, _| {}, &cancel)?;
                project.update(&c.id, |c| c.status = Status::Review);
            }
            project.save()?;
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
