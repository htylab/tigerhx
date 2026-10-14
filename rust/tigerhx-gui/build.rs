fn main() {
    println!("cargo:rerun-if-changed=../../tigerhx/exe/ico.ico");
    println!("cargo:rerun-if-changed=../../tigerhx/webgui/static");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // icon resource 1: the executable's icon, also used for the window
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../tigerhx/exe/ico.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=cannot embed the icon: {e}");
        }
    }
}
