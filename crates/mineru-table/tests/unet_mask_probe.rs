//! Diagnostic probe: what does our UNet actually segment on a real table crop?
//!
//! Not a correctness gate — a measurement. Our wired engine recovers ~32 cells on
//! a page the reference recovers 128 from, and the loss is either in the mask (the
//! neural forward + its preprocessing) or in the classical recovery that reads it.
//! This prints the mask's class histogram, which the reference's own mask can be
//! compared against directly, and the cell count recovery gets out of it — so the
//! two halves can be told apart instead of argued about.
//!
//! ```text
//! MINERU_MODELS_DIR=/path/to/models \
//! MINERU_PROBE_CROP=/path/to/table.png \
//!   cargo test -p mineru-table --release --test unet_mask_probe -- --ignored --nocapture
//! ```

use mineru_burn_common::backend::Cpu;
use mineru_table::unet::model::UnetModel;

#[test]
#[ignore = "diagnostic; needs MINERU_MODELS_DIR + MINERU_PROBE_CROP and a slow CPU forward"]
fn mask_histogram_and_cell_count() {
    let (Ok(crop_path), Ok(_)) = (
        std::env::var("MINERU_PROBE_CROP"),
        std::env::var("MINERU_MODELS_DIR"),
    ) else {
        eprintln!("set MINERU_PROBE_CROP and MINERU_MODELS_DIR");
        return;
    };

    let img = match image::open(&crop_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("could not open {crop_path}: {e}");
            return;
        }
    };
    println!("crop: {}x{}", img.width(), img.height());

    let model = UnetModel::<Cpu>::loaded();
    let mask = match model.segment_mask(&img) {
        Ok(mask) => mask,
        Err(e) => {
            eprintln!("segment_mask failed: {e}");
            return;
        }
    };
    println!("mask: {}x{}", mask.width, mask.height);
    for class in 0..3i64 {
        let n = mask.classes.iter().filter(|c| **c == class).count();
        println!("  class {class}: {n} px");
    }

    match model.segment_cells(&img) {
        Ok(cells) => println!("recovered cells: {}", cells.len()),
        Err(e) => eprintln!("segment_cells failed: {e}"),
    }
}
