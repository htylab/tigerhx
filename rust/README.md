# TigerHx in Rust

    tigerhx-core   port of tigerhx/core: NIfTI, ONNX segmentation, AHA,
                   checks, legacy .mat, workspace, per-case pipeline
    tigerhx-gui    tigerhxgui: the web GUI's backend plus a native window

## Build

    cargo build --release -p tigerhx-gui

`ort` downloads a static ONNX Runtime at build time. Where that download is
blocked, point it at an existing library instead, e.g. the one in the
`onnxruntime` Python wheel (`onnxruntime/capi`, version 1.31 or newer, with a
`libonnxruntime.so` link next to it):

    ORT_LIB_PATH=/path/to/capi ORT_PREFER_DYNAMIC_LINK=1 cargo build

(and `LD_LIBRARY_PATH=/path/to/capi` when running).

The window uses WebView2 on Windows and WebKitGTK on Linux
(`libwebkit2gtk-4.1-dev`, `libgtk-3-dev`). Without the default `webview`
feature the GUI opens in the browser and needs neither:

    cargo build -p tigerhx-gui --no-default-features

## Run

    tigerhxgui [-w WORKSPACE] [--port PORT] [--browser] [--no-open]

Same options and workspace as `tigerhx-gui` (Python). Models are looked up
in `models` next to the executable or in `TIGERHX_MODEL_DIR`, and downloaded
there on first use.

The Windows build is a single file, as in
[rustbx](https://github.com/htylab/rustbx): ONNX Runtime is linked in
statically and the default model is built in when
`rust/models/cine4d_v0006_xy_cine.onnx` exists at build time (or the file named
by `TIGERHX_EMBED_MODEL`; CI downloads it), so it segments without a download.
ONNX Runtime's Windows build links to `DirectML.dll`, which Windows 10 (1903)
and later include; only the CPU is used, so the DLL is not shipped.

## Tests

    cargo test -p tigerhx-core
    cargo build -p tigerhx-core --example parity
    cargo build -p tigerhx-gui
    cd .. && pytest tests/test_rust_parity.py tests/test_rust_server.py

The two Python tests compare the Rust builds with the Python core and server
(they need `pip install -e . onnxruntime onnx pytest`).
