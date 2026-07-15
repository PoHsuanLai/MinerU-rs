//! Diagnostic probe: does table-crop OCR recover its text once the crop is
//! rotated upright?
//!
//! Not a correctness gate — a measurement. Point it at a table crop and it prints
//! what the real det+rec models read at 0/90/270 degrees, so the "rotated tables
//! read as noise" claim can be checked against the models rather than inferred
//! from the assembled document.
//!
//! ```text
//! MINERU_MODELS_DIR=/path/to/PDF-Extract-Kit-1.0/models \
//! MINERU_PROBE_CROP=/path/to/crop.png \
//!   cargo test -p mineru-backend-pipeline --release --test rotation_probe -- --ignored --nocapture
//! ```

use image::RgbImage;
use mineru_burn_common::backend::{cpu_device, Cpu};
use mineru_ocr_det::{DetConfig, TextDetector};
use mineru_ocr_rec::{CharDict, RecConfig, TextRecognizer};

/// Reads every text line the detector finds, in the order it returns them.
fn read_all(det: &TextDetector<Cpu>, rec: &TextRecognizer<Cpu>, img: &RgbImage) -> Vec<String> {
    let boxes = det.detect(img).unwrap_or_default();
    let mut out = Vec::new();
    for b in boxes {
        let x0 = b.x0.max(0.0) as u32;
        let y0 = b.y0.max(0.0) as u32;
        let x1 = (b.x1.min(img.width() as f32) as u32).min(img.width());
        let y1 = (b.y1.min(img.height() as f32) as u32).min(img.height());
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let crop = image::imageops::crop_imm(img, x0, y0, x1 - x0, y1 - y0).to_image();
        if let Ok((text, score)) = rec.recognize(&crop) {
            if !text.trim().is_empty() {
                out.push(format!("{text}  [{score:.2}]"));
            }
        }
    }
    out
}

#[test]
#[ignore = "diagnostic; needs MINERU_MODELS_DIR + MINERU_PROBE_CROP and a slow CPU forward"]
fn ocr_reads_table_crop_at_each_rotation() {
    let (Ok(crop_path), Ok(models_dir)) = (
        std::env::var("MINERU_PROBE_CROP"),
        std::env::var("MINERU_MODELS_DIR"),
    ) else {
        eprintln!("set MINERU_PROBE_CROP and MINERU_MODELS_DIR");
        return;
    };

    let img = image::open(&crop_path)
        .expect("probe crop should open")
        .to_rgb8();
    println!("crop {crop_path}: {}x{}", img.width(), img.height());

    let dev = cpu_device();
    let root = std::path::Path::new(&models_dir);
    let mut det = TextDetector::<Cpu>::new(DetConfig::default(), dev);
    det.load_weights(root.join("OCR/paddleocr_torch/ch_PP-OCRv6_small_det_infer.safetensors"))
        .expect("det weights should load");
    let dict = CharDict::ppocrv6(true).expect("embedded dict should build");
    let mut rec = TextRecognizer::<Cpu>::new(dict, RecConfig::default(), dev);
    rec.load_weights(root.join("OCR/paddleocr_torch/ch_PP-OCRv6_small_rec_infer.safetensors"))
        .expect("rec weights should load");

    for (label, view) in [
        ("0", img.clone()),
        ("90", image::imageops::rotate90(&img)),
        ("270", image::imageops::rotate270(&img)),
    ] {
        let lines = read_all(&det, &rec, &view);
        println!(
            "\n=== rotation {label} ({}x{}) — {} lines read ===",
            view.width(),
            view.height(),
            lines.len()
        );
        for l in lines.iter().take(20) {
            println!("  {l}");
        }
    }
}
