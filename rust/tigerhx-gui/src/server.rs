//! Local HTTP backend for the niivue page (same API as
//! tigerhx/webgui/server.py).
//!
//! Serves the static page, a small JSON API and display volumes for niivue.
//! Bound to 127.0.0.1; the API needs the per-launch token, handed to the page
//! once in the start URL and kept in a SameSite=Strict cookie, so other web
//! pages cannot drive it.

use std::io::Read;
use std::sync::{Arc, OnceLock};

use serde_json::{json, Map, Value};
use tiny_http::{Header, Method, Request, Response, Server};
use tigerhx_core::error::Kind;
use tigerhx_core::pipeline::VolumeKind;
use tigerhx_core::Error;

use crate::app::{truthy, App};

const MAX_BODY: usize = 1 << 20;

/// Native file dialogs: kind is "files" or "folder"; None when cancelled.
pub type Picker = Box<dyn Fn(&str) -> Vec<String> + Send + Sync>;

macro_rules! static_file {
    ($name:literal) => {
        ($name, include_bytes!(concat!("../../../tigerhx/webgui/static/", $name)).as_slice())
    };
}

const STATIC: [(&str, &[u8]); 5] = [
    static_file!("index.html"),
    static_file!("app.css"),
    static_file!("app.js"),
    static_file!("vendor/niivue.umd.js"),
    static_file!("vendor/NIIVUE_LICENSE.txt"),
];

fn content_type(name: &str) -> &'static str {
    match name.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

pub struct Ctx {
    pub app: Arc<App>,
    port: u16,
    pub picker: OnceLock<Picker>,
}

/// One response: status, body, content type and extra headers.
struct Reply {
    code: u16,
    body: Vec<u8>,
    ctype: &'static str,
    headers: Vec<(&'static str, String)>,
}

impl Reply {
    fn new(code: u16, body: impl Into<Vec<u8>>, ctype: &'static str) -> Self {
        Reply { code, body: body.into(), ctype, headers: Vec::new() }
    }
    fn json(code: u16, v: Value) -> Self {
        Reply::new(code, serde_json::to_vec(&v).unwrap(), "application/json")
    }
    fn error(code: u16, msg: impl std::fmt::Display) -> Self {
        Reply::json(code, json!({"error": msg.to_string()}))
    }
    fn from_err(e: Error) -> Self {
        match e.kind {
            Kind::NotFound => Reply::error(404, e),
            Kind::Invalid => Reply::error(400, e),
            _ => {
                log!("TigerHx: {e}");
                Reply::error(500, e)
            }
        }
    }
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).unwrap()
}

/// Constant-time comparison of the token.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (k == key).then_some(v)
    })
}

impl Ctx {
    fn get_header<'r>(req: &'r Request, name: &'static str) -> Option<&'r str> {
        req.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str())
    }

    fn host_ok(&self, req: &Request) -> bool {
        let host = Self::get_header(req, "Host").unwrap_or("");
        host == format!("127.0.0.1:{}", self.port) || host == format!("localhost:{}", self.port)
    }

    fn authed(&self, req: &Request) -> bool {
        let cookie = Self::get_header(req, "Cookie").unwrap_or("");
        let token = cookie.split(';').find_map(|kv| {
            let (k, v) = kv.trim().split_once('=')?;
            (k.trim() == "tigerhx").then(|| v.trim().trim_matches('"'))
        });
        self.host_ok(req) && token.is_some_and(|t| same(t, &self.app.token))
    }

    fn handle(&self, mut req: Request) {
        let reply = match req.method() {
            Method::Get | Method::Head => self.get(&req),
            Method::Post => self.post(&mut req),
            _ => Reply::error(405, "method not allowed"),
        };
        let mut resp = Response::from_data(reply.body)
            .with_status_code(reply.code)
            .with_header(header("Content-Type", reply.ctype))
            .with_header(header("Cache-Control", "no-store"))
            .with_header(header("X-Content-Type-Options", "nosniff"));
        for (k, v) in reply.headers {
            resp.add_header(header(k, &v));
        }
        req.respond(resp).ok();
    }

    fn get(&self, req: &Request) -> Reply {
        let url = req.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if !self.host_ok(req) {
            return Reply::new(403, "forbidden", "text/plain");
        }
        if parts.is_empty() {
            let token = query_param(query, "token").unwrap_or("");
            if !token.is_empty() && same(token, &self.app.token) {
                let mut r = Reply::new(303, "", "text/plain");
                r.headers.push(("Location", "/".into()));
                r.headers.push(("Set-Cookie", format!("tigerhx={}; HttpOnly; SameSite=Strict; Path=/", self.app.token)));
                return r;
            }
            if !self.authed(req) {
                return Reply::new(403, "Open TigerHx from the link printed when it starts.", "text/plain; charset=utf-8");
            }
            return self.static_file("index.html");
        }
        if parts == ["favicon.ico"] {
            return Reply::new(204, "", "image/x-icon");
        }
        if parts[0] == "static" {
            return self.static_file(&parts[1..].join("/"));
        }
        if !self.authed(req) {
            return Reply::error(403, "unauthorized");
        }
        if parts == ["api", "state"] {
            return Reply::json(200, self.app.state());
        }
        if parts.len() == 5 && parts[..2] == ["api", "cases"] && parts[3] == "volume" {
            let kind = VolumeKind::parse(parts[4].split('.').next().unwrap_or(""));
            let (Some(kind), true) = (kind, self.app.project.case(parts[2]).is_some()) else {
                return Reply::error(404, "not found");
            };
            let crop = query_param(query, "crop") == Some("1");
            return match self.app.volume(parts[2], kind, crop) {
                Ok(body) => Reply::new(200, body.as_slice(), "application/octet-stream"),
                Err(e) => Reply::from_err(e),
            };
        }
        Reply::error(404, "not found")
    }

    fn static_file(&self, rel: &str) -> Reply {
        match STATIC.iter().find(|(name, _)| *name == rel) {
            Some((name, body)) => Reply::new(200, *body, content_type(name)),
            None => Reply::new(404, "not found", "text/plain"),
        }
    }

    fn body(req: &mut Request) -> Result<Map<String, Value>, Error> {
        let n = req.body_length().unwrap_or(0);
        if n > MAX_BODY {
            return Err(Error::invalid("request too large"));
        }
        let mut raw = Vec::new();
        req.as_reader().take(MAX_BODY as u64 + 1).read_to_end(&mut raw)?;
        if raw.len() > MAX_BODY {
            return Err(Error::invalid("request too large"));
        }
        if raw.is_empty() {
            return Ok(Map::new());
        }
        match serde_json::from_slice(&raw)? {
            Value::Object(m) => Ok(m),
            _ => Err(Error::invalid("expected a JSON object")),
        }
    }

    fn post(&self, req: &mut Request) -> Reply {
        if !self.authed(req) {
            return Reply::error(403, "unauthorized");
        }
        let url = req.url().to_string();
        let path = url.split('?').next().unwrap_or("");
        let parts: Vec<String> = path.split('/').filter(|s| !s.is_empty()).map(String::from).collect();
        let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
        match Self::body(req).and_then(|body| self.route_post(&parts, &body)) {
            Ok(r) => r,
            Err(e) => Reply::from_err(e),
        }
    }

    fn route_post(&self, parts: &[&str], body: &Map<String, Value>) -> Result<Reply, Error> {
        let app = &self.app;
        Ok(match parts {
            ["api", "cases", "add"] => {
                let added = app.project.add_paths(&strings(body.get("paths"))?, str_of(body, "include"), str_of(body, "exclude"))?;
                Reply::json(200, json!({"added": added.len()}))
            }
            ["api", "cases", "remove"] => {
                app.project.remove(&strings(body.get("ids"))?)?;
                Reply::json(200, json!({"ok": true}))
            }
            ["api", "run"] => Reply::json(200, json!({"queued": app.run(&strings(body.get("ids"))?)})),
            ["api", "stop"] => {
                app.stop();
                Reply::json(200, json!({"ok": true}))
            }
            ["api", "settings"] => {
                app.update_settings(body)?;
                Reply::json(200, json!({"ok": true}))
            }
            ["api", "open"] => {
                let path = app.folder(body.get("id").and_then(Value::as_str));
                if path.is_dir() {
                    open_in_file_manager(&path);
                }
                Reply::json(200, json!({"path": path.to_string_lossy()}))
            }
            ["api", "pick"] => match self.picker.get() {
                Some(pick) => {
                    let kind = if str_of(body, "kind") == "folder" { "folder" } else { "files" };
                    Reply::json(200, json!({"paths": pick(kind)}))
                }
                None => Reply::error(404, "no native dialogs"),
            },
            ["api", "cases", id, "apex"] => {
                if app.project.case(id).is_none() {
                    return Ok(Reply::error(404, "not found"));
                }
                let apex = body.get("apex_slice").ok_or_else(|| Error::invalid("'apex_slice'"))?;
                let apex = as_int(apex).ok_or_else(|| Error::invalid(format!("invalid apex slice: {apex}")))?;
                let base_first = body.get("base_first").filter(|v| !v.is_null()).map(truthy);
                app.set_apex(id, apex, base_first)?;
                Reply::json(200, json!({"ok": true}))
            }
            _ => Reply::error(404, "not found"),
        })
    }
}

fn str_of<'a>(body: &'a Map<String, Value>, key: &str) -> &'a str {
    body.get(key).and_then(Value::as_str).unwrap_or("")
}

/// A list of strings (missing or null = empty).
fn strings(v: Option<&Value>) -> Result<Vec<String>, Error> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| x.as_str().map(String::from).ok_or_else(|| Error::invalid("expected a list of strings")))
            .collect(),
        Some(_) => Err(Error::invalid("expected a list of strings")),
    }
}

/// Python's int() of a JSON number or numeric string.
fn as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

pub fn open_in_file_manager(path: &std::path::Path) {
    let program = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    if let Err(e) = std::process::Command::new(program).arg(path).spawn() {
        log!("TigerHx: cannot open {}: {e}", path.display());
    }
}

/// Start the server on background threads; returns the context and the
/// start URL (with the token).
pub fn serve(app: Arc<App>, port: u16) -> std::io::Result<(Arc<Ctx>, String)> {
    let server = Server::http(("127.0.0.1", port)).map_err(std::io::Error::other)?;
    let port = server.server_addr().to_ip().map(|a| a.port()).unwrap_or(port);
    let url = format!("http://127.0.0.1:{port}/?token={}", app.token);
    let ctx = Arc::new(Ctx { app, port, picker: OnceLock::new() });
    let c = ctx.clone();
    std::thread::spawn(move || {
        for req in server.incoming_requests() {
            let c = c.clone();
            std::thread::spawn(move || c.handle(req));
        }
    });
    Ok((ctx, url))
}
