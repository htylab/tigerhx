//! TigerHx GUI launcher.
//!
//! Runs the local backend and shows the page in a native window (or, with
//! --browser or without the `webview` feature, in the default browser).

#![cfg_attr(all(windows, feature = "webview", not(debug_assertions)), windows_subsystem = "windows")]

/// eprintln! that never panics (stderr may be a closed pipe or absent).
macro_rules! log {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

mod app;
mod server;
#[cfg(feature = "webview")]
mod window;

use std::path::PathBuf;

const USAGE: &str = "\
usage: tigerhxgui [-w WORKSPACE] [--port PORT] [--browser] [--no-open]

TigerHx cine MRI segmentation GUI

options:
  -w, --workspace WORKSPACE  folder for results and the case list (default: ~/TigerHx)
  --port PORT                local port (default: random)
  --browser                  use the default web browser instead of a native window
  --no-open                  only print the URL
  -h, --help                 show this help";

struct Args {
    workspace: PathBuf,
    port: u16,
    browser: bool,
    no_open: bool,
}

fn default_workspace() -> PathBuf {
    if let Some(w) = std::env::var_os("TIGERHX_WORKSPACE").filter(|w| !w.is_empty()) {
        return PathBuf::from(w);
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).unwrap_or_default();
    PathBuf::from(home).join("TigerHx")
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args { workspace: default_workspace(), port: 0, browser: false, no_open: false };
    let mut it = std::env::args_os().skip(1);
    while let Some(a) = it.next() {
        let a = a.to_string_lossy().into_owned();
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let mut value = |name: &str| -> Result<std::ffi::OsString, String> {
            inline.clone().map(Into::into).or_else(|| it.next()).ok_or(format!("{name} needs a value"))
        };
        match flag.as_str() {
            "-w" | "--workspace" => args.workspace = PathBuf::from(value(&flag)?),
            "--port" => {
                let v = value(&flag)?;
                args.port = v.to_string_lossy().parse().map_err(|_| format!("invalid port {v:?}"))?;
            }
            "--browser" => args.browser = true,
            "--no-open" => args.no_open = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized argument {other}")),
        }
    }
    Ok(args)
}

pub fn open_browser(url: &str) {
    let result = if cfg!(windows) {
        // `start` treats the first quoted argument as the window title
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if let Err(e) = result {
        log!("TigerHx: cannot open the browser ({e}); open {url}");
    }
}

pub fn wait_forever() -> ! {
    loop {
        std::thread::park();
    }
}

/// A windowed (no console) build started from a terminal prints there.
fn attach_console() {
    #[cfg(all(windows, feature = "webview", not(debug_assertions)))]
    unsafe {
        use windows_sys::Win32::System::Console::{AttachConsole, GetStdHandle, ATTACH_PARENT_PROCESS, STD_OUTPUT_HANDLE};
        if GetStdHandle(STD_OUTPUT_HANDLE).is_null() {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

fn main() {
    attach_console();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            log!("{USAGE}\ntigerhxgui: error: {e}");
            std::process::exit(2);
        }
    };
    let app = match app::App::new(&args.workspace) {
        Ok(a) => a,
        Err(e) => {
            log!("tigerhxgui: cannot open the workspace {}: {e}", args.workspace.display());
            std::process::exit(1);
        }
    };
    let (ctx, url) = match server::serve(app.clone(), args.port) {
        Ok(s) => s,
        Err(e) => {
            log!("tigerhxgui: cannot start the local server: {e}");
            std::process::exit(1);
        }
    };
    println!("TigerHx workspace: {}", app.project.workspace.display());
    println!("TigerHx GUI: {url}");

    #[cfg(feature = "webview")]
    if !args.browser && !args.no_open {
        let a = app.clone();
        window::run(ctx, url, move || a.stop());
    }
    let _ = &ctx;
    if !args.no_open {
        open_browser(&url);
    }
    wait_forever();
}
