//! Diagnostic probe: where does a formula decode actually spend its time?
//!
//! Not a correctness gate — a measurement. The decode is ~87% of pipeline
//! wall-clock (245s of 284s on a 15-page paper), while the reference does the same
//! 89 crops in ~12s. Both emit the same LaTeX at the same lengths, so the gap is
//! cost per decoder step, not extra work. This splits a real decode by batch width
//! to find where the per-crop cost stops falling — that bounds what more batching
//! can buy and says whether the remaining gap is elsewhere.
//!
//! ```text
//! MINERU_FORMULA_MODEL_DIR=/path/to/unimernet_hf_small_2503 \
//! MINERU_FORMULA_CROP=/path/to/crop.raw \
//!   cargo test -p mineru-formula --release --test decode_profile -- --ignored --nocapture
//! ```

use mineru_burn_common::backend::Cpu;
use std::time::Instant;

/// Reads the `w:u32 | h:u32 | rgb bytes` dump the other real-weights probes use.
fn read_raw_crop(path: &str) -> Option<image::RgbImage> {
    let bytes = std::fs::read(path).ok()?;
    let w = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
    let h = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?);
    image::RgbImage::from_raw(w, h, bytes.get(8..)?.to_vec())
}

#[test]
#[ignore = "diagnostic; needs the checkpoint + a crop, and a slow CPU forward"]
fn decode_cost_by_batch_width() {
    use mineru_burn_common::weights::Coverage;
    use mineru_formula::FormulaRecognizer;

    let (Ok(dir), Ok(crop_path)) = (
        std::env::var("MINERU_FORMULA_MODEL_DIR"),
        std::env::var("MINERU_FORMULA_CROP"),
    ) else {
        eprintln!("set MINERU_FORMULA_MODEL_DIR and MINERU_FORMULA_CROP");
        return;
    };

    let t = Instant::now();
    let recognizer = match FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("load failed: {e}");
            return;
        }
    };
    println!("load: {:.2}s", t.elapsed().as_secs_f32());

    let Some(img) = read_raw_crop(&crop_path) else {
        eprintln!("could not read crop at {crop_path}");
        return;
    };
    println!("crop: {}x{}", img.width(), img.height());

    // Per-token cost is the number that matters: a slow decode and a long one look
    // identical from a total.
    let t = Instant::now();
    let tokens = match recognizer.predict_token_ids(&img) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("predict failed: {e}");
            return;
        }
    };
    let solo = t.elapsed().as_secs_f32();
    println!(
        "scalar predict: {:.2}s / {} tokens = {:.1} ms/token",
        solo,
        tokens.len(),
        solo * 1000.0 / tokens.len().max(1) as f32,
    );

    // Batching amortizes the decoder's weight reads across lanes. Where per-crop
    // cost stops falling is where the decode stops being bandwidth-bound.
    for lanes in [1usize, 2, 4, 8, 16] {
        let crops: Vec<image::RgbImage> = (0..lanes).map(|_| img.clone()).collect();
        let t = Instant::now();
        let out = match recognizer.predict_batch(&crops) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("batch {lanes} failed: {e}");
                continue;
            }
        };
        let elapsed = t.elapsed().as_secs_f32();
        let done = out.iter().filter(|o| o.is_some()).count();
        println!(
            "batch {lanes:2}: {elapsed:6.2}s total  {:5.2}s/crop  ({done}/{lanes} decoded)",
            elapsed / lanes as f32,
        );
    }
}
