//! Build script: optional ONNX -> Burn codegen for the LCNet classifier and the
//! UNet line-segmentation model.
//!
//! Codegen only runs when the `onnx-import` cargo feature is enabled *and* the
//! `.onnx` files are found on disk. When either condition is false the script is
//! a no-op, and the library compiles model-free stubs (see `src/cls.rs` and
//! `src/unet/model.rs`). This keeps the crate buildable with no model files and
//! no network access, which is the default.
//!
//! Model files are looked up under `$MINERU_MODELS_DIR` using the same relative
//! paths as the Python `ModelPath` enum:
//!   - `models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx`
//!   - `models/TabRec/UnetStructure/unet.onnx`
//!
//! An unsupported ONNX op makes burn-onnx's `ModelGen` panic at build time; if
//! that happens, disable the `onnx-import` feature and fall back to the stub for
//! the offending model (and please report which op failed).

fn main() {
    println!("cargo:rerun-if-env-changed=MINERU_MODELS_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    // Advertise cfgs the library uses to switch between generated code and stubs,
    // so `cargo` does not warn about unexpected `cfg` values.
    println!("cargo::rustc-check-cfg=cfg(lcnet_generated)");
    println!("cargo::rustc-check-cfg=cfg(unet_generated)");

    #[cfg(feature = "onnx-import")]
    generate();
}

#[cfg(feature = "onnx-import")]
fn generate() {
    use std::path::PathBuf;

    let models_dir = match std::env::var_os("MINERU_MODELS_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            println!(
                "cargo:warning=onnx-import enabled but MINERU_MODELS_DIR is unset; \
                 skipping ONNX codegen and compiling model stubs."
            );
            return;
        }
    };

    let lcnet = models_dir.join("models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx");
    let unet = models_dir.join("models/TabRec/UnetStructure/unet.onnx");

    if lcnet.exists() {
        burn_onnx::ModelGen::new()
            .input(lcnet.to_string_lossy().as_ref())
            .out_dir("model/")
            .run_from_script();
        println!("cargo:rustc-cfg=lcnet_generated");
    } else {
        println!(
            "cargo:warning=LCNet ONNX not found at {}; compiling classifier stub.",
            lcnet.display()
        );
    }

    if unet.exists() {
        burn_onnx::ModelGen::new()
            .input(unet.to_string_lossy().as_ref())
            .out_dir("model/")
            .run_from_script();
        println!("cargo:rustc-cfg=unet_generated");
    } else {
        println!(
            "cargo:warning=UNet ONNX not found at {}; compiling segmentation stub.",
            unet.display()
        );
    }
}
