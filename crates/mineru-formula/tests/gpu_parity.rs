//! GPU-vs-CPU parity regression test for the formula model.
//!
//! Guards the fix for a Burn 0.21 wgpu matmul bug (`M >= 512 && K >= 512` returns
//! wrong results — see [`mineru_burn_common::nn::linear_tiled`]) that corrupted the
//! Swin encoder's FFN and produced garbled formula LaTeX on GPU. This runs BOTH
//! backends on the identical preprocessed pixel tensor and asserts the encoder grid
//! and the decoder logits match to floating-point noise, stage by stage, so a
//! regression re-localizes immediately to the encoder or the decoder.
//!
//! Requires the `gpu` feature and a Metal/Vulkan device.
//!
//! ```text
//! MINERU_FORMULA_DIR=/path/unimernet_hf_small_2503 \
//!   cargo test -p mineru-formula --release --features gpu \
//!     --test gpu_parity -- --ignored --nocapture
//! ```

#![cfg(feature = "gpu")]

use std::path::Path;

/// Read a Python-dumped `<name>.bin` (little-endian f32) + `<name>.shape`.
fn read_ref(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let bin = std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("read {name}.bin: {e}"));
    let shape: Vec<usize> = std::fs::read_to_string(dir.join(format!("{name}.shape")))
        .unwrap_or_else(|e| panic!("read {name}.shape: {e}"))
        .trim()
        .split(',')
        .map(|s| s.parse().expect("shape dim"))
        .collect();
    let vals: Vec<f32> = bin
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (vals, shape)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut idx = 0;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > m {
            m = d;
            idx = i;
        }
    }
    (m, idx)
}

#[test]
#[ignore = "needs the gpu feature + a GPU + the checkpoint dir"]
fn cpu_vs_gpu_stage_divergence() {
    use burn::tensor::{Tensor, TensorData};
    use mineru_burn_common::backend::{cpu_device, gpu_available, gpu_device, Cpu, Gpu};
    use mineru_burn_common::weights::Coverage;
    use mineru_formula::FormulaRecognizer;

    let Ok(dir) = std::env::var("MINERU_FORMULA_DIR") else {
        eprintln!("set MINERU_FORMULA_DIR to the unimernet_hf_small_2503 dir");
        return;
    };
    if !gpu_available() {
        eprintln!("no usable GPU; skipping");
        return;
    }

    // Both recognizers load the SAME weights from disk; only the backend differs.
    let rec_cpu = FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict)
        .expect("cpu load");
    let rec_gpu = FormulaRecognizer::<Gpu>::from_pretrained_on(&dir, Coverage::Strict, gpu_device())
        .expect("gpu load");

    // The real preprocessed formula image the parity fixtures were dumped from.
    let ref_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("reference");
    let (px, ps) = read_ref(&ref_dir, "input");
    let pix_cpu: Tensor<Cpu, 4> =
        Tensor::from_data(TensorData::new(px.clone(), [ps[0], ps[1], ps[2], ps[3]]), &cpu_device());
    let pix_gpu: Tensor<Gpu, 4> =
        Tensor::from_data(TensorData::new(px, [ps[0], ps[1], ps[2], ps[3]]), &gpu_device());

    // ---- Stage 1: the FINAL encoder grid [1, L, d] (the decoder's input) ----
    // `encode_stages` returns (patch_embed, [stage outputs]); the final grid the
    // decoder consumes is the last stage.
    let (embed_cpu, stages_cpu) = rec_cpu.encode_stages(pix_cpu);
    let (embed_gpu, stages_gpu) = rec_gpu.encode_stages(pix_gpu.clone());

    // Walk the Swin pipeline stage by stage to find where wgpu first diverges.
    println!("\n=== Swin stage-by-stage CPU vs GPU ===");
    let ecv = mineru_burn_common::float_to_vec_f32(embed_cpu.clone());
    let egv = mineru_burn_common::float_to_vec_f32(embed_gpu.clone());
    let (em, _) = max_abs_diff(&ecv, &egv);
    println!("patch_embed {:?}: max|Δ| {em:.5}", embed_cpu.dims());
    for (i, (sc, sg)) in stages_cpu.iter().zip(&stages_gpu).enumerate() {
        let a = mineru_burn_common::float_to_vec_f32(sc.clone());
        let b = mineru_burn_common::float_to_vec_f32(sg.clone());
        let (m, idx) = max_abs_diff(&a, &b);
        println!("stage {i} {:?}: max|Δ| {m:.5} at {idx}", sc.dims());
    }

    let enc_cpu = stages_cpu.last().expect("cpu stages").clone();
    let enc_gpu = stages_gpu.last().expect("gpu stages").clone();
    let enc_cpu_v = mineru_burn_common::float_to_vec_f32(enc_cpu.clone());
    let enc_gpu_v = mineru_burn_common::float_to_vec_f32(enc_gpu.clone());
    let (enc_max, enc_idx) = max_abs_diff(&enc_cpu_v, &enc_gpu_v);
    let d = enc_cpu.dims();
    println!("\n=== encoder grid [1,L,d] ===");
    println!("shape {d:?}, max|Δ| {enc_max:.5} at {enc_idx}");

    let bos = 0u32; // decoder start token (config bos_token_id = 0)

    // ---- Stage 2: first-step decoder logits, each on its OWN encoder output ----
    let logits_cpu = rec_cpu.decoder_step_logits(&[bos], enc_cpu.clone());
    let logits_gpu = rec_gpu.decoder_step_logits(&[bos], enc_gpu.clone());
    let lc = mineru_burn_common::float_to_vec_f32(logits_cpu);
    let lg = mineru_burn_common::float_to_vec_f32(logits_gpu);
    let (lmax, lidx) = max_abs_diff(&lc, &lg);
    let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
    let (am_cpu, am_gpu) = (am(&lc), am(&lg));
    println!("\n=== step-0 decoder logits [1,vocab] (each on its own encoder) ===");
    println!("max|Δ| {lmax:.4} at {lidx}; argmax cpu={am_cpu} gpu={am_gpu} {}",
        if am_cpu == am_gpu { "(SAME token)" } else { "*** DIFFERENT TOKEN ***" });

    // ---- Stage 2b: GPU decoder fed the CPU encoder grid (cross-fed) ----
    // Isolates decoder vs encoder: if this matches CPU, the encoder diverged; if it
    // still differs, the decoder itself diverges on wgpu.
    let enc_cpu_on_gpu: Tensor<Gpu, 3> =
        Tensor::from_data(TensorData::new(enc_cpu_v, [d[0], d[1], d[2]]), &gpu_device());
    let logits_gpu_cpuenc = rec_gpu.decoder_step_logits(&[bos], enc_cpu_on_gpu);
    let lg2 = mineru_burn_common::float_to_vec_f32(logits_gpu_cpuenc);
    let (lmax2, _) = max_abs_diff(&lc, &lg2);
    let am_gpu2 = am(&lg2);
    println!("\n=== GPU decoder fed the CPU encoder grid (cross-fed) ===");
    println!("max|Δ| vs CPU {lmax2:.4}; argmax gpu(cpu-enc)={am_gpu2} vs cpu={am_cpu} {}",
        if am_cpu == am_gpu2 { "(SAME → the ENCODER is the culprit)" } else { "*** STILL DIFF → the DECODER is the culprit ***" });

    // Regression asserts. The tolerances are generous fp-noise bounds; the bug this
    // guards produced max|Δ| in the tens, so the gap is unmistakable.
    assert!(enc_max < 1e-2, "encoder grid diverged on GPU (max|Δ| {enc_max}) — wgpu matmul regression?");
    assert!(lmax < 1.0, "decoder logits diverged on GPU (max|Δ| {lmax})");
    assert_eq!(am_cpu, am_gpu, "GPU argmax flipped vs CPU");
}
