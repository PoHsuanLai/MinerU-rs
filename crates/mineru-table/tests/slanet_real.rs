//! Hard numeric parity gate for the hand-ported SLANet-plus decoder.
//!
//! This test is the un-fakeable proof that the Rust autoregressive decoder
//! reproduces the real `slanet-plus.onnx` `Loop` — not just a shape-correct
//! forward. It runs the model on the SAME deterministic 488×488 4×4 grid that
//! `tests/reference/py_ref_slanet.py` fed to `onnxruntime`, then asserts, against
//! the committed reference dumps:
//!
//! 1. the per-step structure **argmax** sequence equals the ONNX token stream
//!    `[5,48,48,48,48,6] × 4, 49, 0` (four `<tr> <td></td>×4 </tr>` rows, then eos);
//! 2. the structure **probabilities** match `slanet_structure.bin` to `< 1e-3`;
//! 3. the **loc quad** corners match `slanet_loc.bin` to `< 1e-2`.
//!
//! A prior port inserted a spurious ReLU in the structure/box heads; it matched
//! steps 0–3 then flipped token 6↔48 at step 4. This gate catches exactly that
//! class of bug.
//!
//! The `.bin`/`.shape` dumps are gitignored (regenerate with the venv:
//! `python tests/reference/py_ref_slanet.py`). This test is `#[ignore]`d because
//! it needs the model weights on disk and the slow ndarray CPU forward.
//!
//! Run with:
//!
//! ```text
//! SLANET_ONNX=/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/SlanetPlus/slanet-plus.onnx \
//!   cargo test -p mineru-table --test slanet_real -- --ignored --nocapture
//! ```

use image::{Rgb, RgbImage};
use mineru_table::slanet::model::SlaNet;
use mineru_table::slanet::preprocess::preprocess;
use mineru_table::slanet::{build_vocab, decode::decode};

/// Default model path (overridable via `SLANET_ONNX`).
const DEFAULT_ONNX: &str =
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/SlanetPlus/slanet-plus.onnx";

/// The ONNX ground-truth structure argmax sequence on the reference grid:
/// four `<tr> <td></td>×4 </tr>` rows, then eos (49) and the trailing padding 0.
const ORT_ARGMAX: [usize; 26] = [
    5, 48, 48, 48, 48, 6, 5, 48, 48, 48, 48, 6, 5, 48, 48, 48, 48, 6, 5, 48, 48, 48, 48, 6, 49, 0,
];

/// Number of structure class channels.
const NUM_CLASSES: usize = 50;
/// Loc quad width (four `(x, y)` corners).
const LOC_DIM: usize = 8;

/// Builds the deterministic 4×4 grid the reference dumper uses.
fn synthetic_table(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    let sh = (h / 4).max(1);
    let sw = (w / 4).max(1);
    let mut y = 0;
    while y < h {
        for x in 0..w {
            img.put_pixel(x, y.min(h - 1), Rgb([0, 0, 0]));
        }
        y += sh;
    }
    let mut x = 0;
    while x < w {
        for yy in 0..h {
            img.put_pixel(x.min(w - 1), yy, Rgb([0, 0, 0]));
        }
        x += sw;
    }
    img
}

/// Loads a committed little-endian f32 reference dump, or returns `None` (with a
/// note) when the regenerable `.bin` is absent so the test can skip cleanly.
fn load_ref(name: &str) -> Option<Vec<f32>> {
    let path = format!("{}/tests/reference/{name}.bin", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Index of the maximum element (first on ties).
fn argmax(row: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

#[test]
#[ignore = "requires the SLANet-plus model files + reference dumps on disk; slow ndarray forward"]
fn slanet_matches_onnx_reference() {
    let onnx = std::env::var("SLANET_ONNX").unwrap_or_else(|_| DEFAULT_ONNX.to_string());

    let ref_struct = match load_ref("slanet_structure") {
        Some(v) => v,
        None => {
            eprintln!(
                "SKIP: tests/reference/slanet_structure.bin missing; \
                 regenerate with `python tests/reference/py_ref_slanet.py`"
            );
            return;
        }
    };
    let ref_loc = load_ref("slanet_loc").expect("slanet_loc.bin must accompany slanet_structure.bin");
    let ref_len = ref_struct.len() / NUM_CLASSES;
    assert_eq!(ref_struct.len(), ref_len * NUM_CLASSES, "ref structure must be [L,50]");
    assert_eq!(ref_loc.len(), ref_len * LOC_DIM, "ref loc must be [L,8]");
    assert_eq!(ref_len, ORT_ARGMAX.len(), "reference should be 26 steps");

    let model = SlaNet::<mineru_burn_common::backend::Cpu>::load(&onnx).expect("SlaNet::load should succeed");
    let img = synthetic_table(488, 488);
    let pre = preprocess(&img);

    let (probs, quads, len) = model
        .debug_raw_head(&pre)
        .expect("forward must run with real weights");
    println!("Rust decoded {len} steps (reference has {ref_len})");
    assert!(len > 0, "decoded length must be positive");

    // The Rust decoder greedily stops one step after emitting eos; compare over the
    // steps it produced, which must be a prefix of the ONNX sequence.
    let cmp = len.min(ref_len);
    assert!(cmp >= 25, "should decode the full 4-row table before eos, got {len}");

    // 1) Argmax sequence parity.
    let rust_arg: Vec<usize> = (0..cmp)
        .map(|s| argmax(&probs[s * NUM_CLASSES..(s + 1) * NUM_CLASSES]))
        .collect();
    for (s, (&r, &o)) in rust_arg.iter().zip(ORT_ARGMAX.iter()).enumerate() {
        assert_eq!(
            r, o,
            "structure argmax mismatch at step {s}: rust={r} onnx={o}\n\
             rust seq so far: {:?}",
            &rust_arg[..=s]
        );
    }
    println!("argmax parity OK over {cmp} steps: {rust_arg:?}");

    // 2) Structure probability parity (< 1e-3 max-abs).
    let mut max_p = 0.0f32;
    for i in 0..cmp * NUM_CLASSES {
        max_p = max_p.max((probs[i] - ref_struct[i]).abs());
    }
    println!("structure probs max-abs diff = {max_p:.3e}");
    assert!(max_p < 1e-3, "structure prob max-abs diff {max_p:.3e} exceeds 1e-3");

    // 3) Loc quad parity (< 1e-2 max-abs).
    let mut max_l = 0.0f32;
    for i in 0..cmp * LOC_DIM {
        max_l = max_l.max((quads[i] - ref_loc[i]).abs());
    }
    println!("loc quad max-abs diff = {max_l:.3e}");
    assert!(max_l < 1e-2, "loc quad max-abs diff {max_l:.3e} exceeds 1e-2");

    // Sanity: the public decode path yields a well-formed 4-row table.
    let out = model.forward(&pre).expect("forward");
    let vocab = build_vocab();
    let res = decode(&out.as_preds(), &vocab, pre.orig_w, pre.orig_h);
    let tr_count = res.tokens.iter().filter(|t| *t == "<tr>").count();
    let td_count = res.tokens.iter().filter(|t| *t == "<td></td>").count();
    println!("decoded {tr_count} rows, {td_count} cells; score={:.3}", res.score);
    assert_eq!(tr_count, 4, "expected 4 table rows");
    assert_eq!(td_count, 16, "expected 16 cells (4x4 grid)");
}
