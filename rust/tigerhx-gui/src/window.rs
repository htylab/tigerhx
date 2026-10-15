//! The native window: the page in a webview (WebView2 on Windows, WebKitGTK
//! on Linux, WKWebView on macOS) plus native file dialogs.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::WindowBuilder;
use wry::{NewWindowResponse, WebContext, WebViewBuilder};

use crate::server::Ctx;

/// The page asks for file dialogs through window.tigerhxNative; the server
/// passes the request to the window's thread (see /api/pick).
const SHIM: &str = r#"
window.tigerhxNative = {
  pick(kind) {
    return fetch('/api/pick', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind }),
    }).then(r => r.json()).then(d => d.paths || []);
  },
  pick_files() { return this.pick('files'); },
  pick_folder() { return this.pick('folder'); },
};
"#;

enum UserEvent {
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    Pick(String, mpsc::Sender<Vec<String>>),
}

fn pick(dialog: rfd::FileDialog, kind: &str) -> Vec<String> {
    let paths = if kind == "folder" {
        dialog.pick_folder().map(|p| vec![p])
    } else {
        dialog
            .add_filter("NIfTI, case list or result", &["nii", "gz", "csv", "mat"])
            .add_filter("NIfTI (*.nii, *.nii.gz)", &["nii", "gz"])
            .add_filter("Case list (*.csv)", &["csv"])
            .add_filter("TigerHx result (*.mat)", &["mat"])
            .add_filter("All files", &["*"])
            .pick_files()
    };
    paths.unwrap_or_default().into_iter().map(|p| p.to_string_lossy().into_owned()).collect()
}

/// Where the webview keeps its profile (not next to the executable, which
/// may be read-only).
fn data_dir() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("TigerHx").join("webview"))
}

fn window_icon() -> Option<tao::window::Icon> {
    #[cfg(windows)]
    {
        use tao::platform::windows::IconExtWindows;
        return tao::window::Icon::from_resource(1, None).ok();
    }
    #[allow(unreachable_code)]
    None
}

/// Show the page; returns when the window is closed (after `on_close`).
pub fn run(ctx: Arc<Ctx>, url: String, on_close: impl FnOnce() + 'static) -> ! {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    // GTK (Linux) runs the dialog on its main loop, so the request thread
    // opens it and waits; elsewhere the window's thread opens it, modal.
    #[cfg(target_os = "linux")]
    let _ = ctx.picker.set(Box::new(|kind: &str| pick(rfd::FileDialog::new(), kind)));
    #[cfg(not(target_os = "linux"))]
    {
        let proxy = std::sync::Mutex::new(event_loop.create_proxy());
        let _ = ctx.picker.set(Box::new(move |kind: &str| {
            let (tx, rx) = mpsc::channel();
            if proxy.lock().unwrap().send_event(UserEvent::Pick(kind.to_string(), tx)).is_err() {
                return Vec::new();
            }
            rx.recv().unwrap_or_default()
        }));
    }

    let window = WindowBuilder::new()
        .with_title("TigerHx")
        .with_inner_size(LogicalSize::new(1440.0, 900.0))
        .with_min_inner_size(LogicalSize::new(1000.0, 640.0))
        .with_window_icon(window_icon())
        .build(&event_loop)
        .expect("cannot create the window");

    let origin = url.split("/?").next().unwrap_or(&url).to_string();
    let mut web_context = WebContext::new(data_dir());
    let builder = WebViewBuilder::new_with_web_context(&mut web_context)
        .with_url(&url)
        .with_initialization_script(SHIM)
        .with_devtools(cfg!(debug_assertions))
        // stay on the local page; nothing opens new windows
        .with_navigation_handler(move |to| to.starts_with(&format!("{origin}/")) || to == origin)
        .with_new_window_req_handler(|_, _| NewWindowResponse::Deny);
    // a debugging port for tools/ui_test_exe.py (wry's default arguments kept)
    #[cfg(windows)]
    let builder = match std::env::var("TIGERHX_WEBVIEW_DEBUG_PORT") {
        Ok(port) if !port.is_empty() => {
            use wry::WebViewBuilderExtWindows;
            builder.with_additional_browser_args(format!(
                "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --remote-debugging-port={port}"
            ))
        }
        _ => builder,
    };
    #[cfg(not(target_os = "linux"))]
    let webview = builder.build(&window);
    #[cfg(target_os = "linux")]
    let webview = {
        use tao::platform::unix::WindowExtUnix;
        use wry::WebViewBuilderExtUnix;
        builder.build_gtk(window.default_vbox().expect("no GTK box"))
    };
    let webview = match webview {
        Ok(w) => w,
        Err(e) => {
            log!("TigerHx: cannot create the web view ({e}); opening the browser instead");
            crate::open_browser(&url);
            crate::wait_forever();
        }
    };

    let mut on_close = Some(on_close);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let _keep = (&webview, &web_context);
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                if let Some(f) = on_close.take() {
                    f();
                }
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(UserEvent::Pick(kind, reply)) => {
                reply.send(pick(rfd::FileDialog::new().set_parent(&window), &kind)).ok();
            }
            _ => {}
        }
    })
}
