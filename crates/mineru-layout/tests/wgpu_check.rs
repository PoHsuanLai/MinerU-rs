//! wgpu-vs-CPU parity for the layout model, gated behind `#[ignore]` and the
//! `gpu` feature so the default test run never needs a GPU toolchain or weights.
//!
//! Run with the checkpoint present and a Metal/Vulkan GPU available:
//!
//! ```sh
//! MINERU_LAYOUT_WEIGHTS=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/Layout/PP-DocLayoutV2/model.safetensors \
//!     cargo test -p mineru-layout --features gpu --test wgpu_check -- --ignored --nocapture
//! ```
//!
//! The test loads the *same* weights on both the CPU (`Flex`) and GPU (`Wgpu`)
//! backends, runs `detect` on both, and asserts the detections match: same count,
//! same labels, same reading order, and boxes/scores within a small tolerance.
//! This is the proof that the on-device threshold-sort/reading-order rewrites (and
//! the dtype-agnostic host reads) produce identical results on wgpu — where int
//! tensors are stored as `i32`, not `i64`.

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use image::{Rgb, RgbImage};
use mineru_burn_common::backend::{cpu_device, gpu_device, Cpu, Gpu};
use mineru_layout::LayoutModel;

fn weights_path() -> Option<PathBuf> {
    std::env::var_os("MINERU_LAYOUT_WEIGHTS").map(PathBuf::from)
}

/// Builds a deterministic non-blank test page: a few filled rectangles on white,
/// so the model actually produces detections to compare (a blank page yields an
/// empty list, which would pass trivially).
fn test_page() -> RgbImage {
    let mut img = RgbImage::from_pixel(1000, 1400, Rgb([255, 255, 255]));
    // A "title" bar near the top and two "text" blocks below.
    for (x0, y0, x1, y1, shade) in [
        (100u32, 80u32, 900u32, 160u32, 40u8),
        (100, 240, 900, 620, 90),
        (100, 700, 900, 1080, 90),
    ] {
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, Rgb([shade, shade, shade]));
            }
        }
    }
    img
}

#[test]
#[ignore = "requires the PP-DocLayoutV2 safetensors checkpoint and a GPU"]
fn wgpu_detect_matches_cpu() {
    let Some(path) = weights_path() else {
        eprintln!("MINERU_LAYOUT_WEIGHTS not set; skipping");
        return;
    };

    let page = test_page();

    let cpu = LayoutModel::<Cpu>::load(&path, cpu_device()).expect("load CPU weights");
    let cpu_dets = cpu.detect(&page).expect("CPU detect");
    eprintln!("CPU detections: {}", cpu_dets.len());
    for d in &cpu_dets {
        eprintln!(
            "  CPU o={} {:?} score={:.5} bbox=[{:.2},{:.2},{:.2},{:.2}]",
            d.order, d.label, d.score, d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1
        );
    }

    let gpu = LayoutModel::<Gpu>::load(&path, gpu_device()).expect("load GPU weights");
    let gpu_dets = gpu.detect(&page).expect("GPU detect");
    eprintln!("GPU detections: {}", gpu_dets.len());
    for d in &gpu_dets {
        eprintln!(
            "  GPU o={} {:?} score={:.5} bbox=[{:.2},{:.2},{:.2},{:.2}]",
            d.order, d.label, d.score, d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1
        );
    }

    assert_eq!(
        cpu_dets.len(),
        gpu_dets.len(),
        "detection count differs: cpu={} gpu={}",
        cpu_dets.len(),
        gpu_dets.len()
    );

    for (i, (c, g)) in cpu_dets.iter().zip(gpu_dets.iter()).enumerate() {
        assert_eq!(c.order, g.order, "det {i}: reading order differs");
        assert_eq!(c.label, g.label, "det {i}: label differs ({:?} vs {:?})", c.label, g.label);
        assert!(
            (c.score - g.score).abs() < 1e-2,
            "det {i}: score differs cpu={} gpu={}",
            c.score,
            g.score
        );
        // Boxes are in PIXEL coordinates on a 1000x1400 page. wgpu and ndarray
        // accumulate f32 in different orders (tiled-2D vs matrixmultiply kernels),
        // so raw coordinates differ by a fraction of a pixel — expected and not a
        // decision difference. The bar is a sane pixel tolerance (2 px ≈ 0.2% of the
        // page), NOT bit-exactness. What must match exactly is the *decisions*: the
        // count, labels, and reading order (asserted above).
        const BOX_TOL_PX: f32 = 2.0;
        let (cb, gb) = (&c.bbox, &g.bbox);
        for (name, cv, gv) in [
            ("x0", cb.x0, gb.x0),
            ("y0", cb.y0, gb.y0),
            ("x1", cb.x1, gb.x1),
            ("y1", cb.y1, gb.y1),
        ] {
            assert!(
                (cv - gv).abs() < BOX_TOL_PX,
                "det {i}: bbox.{name} differs by >{BOX_TOL_PX}px cpu={cv} gpu={gv}"
            );
        }
    }

    eprintln!("wgpu-vs-CPU parity OK: {} detections match (labels/order/count identical, boxes within 2px)", cpu_dets.len());
}
