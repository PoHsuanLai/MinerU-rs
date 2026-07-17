//! GPU-vs-CPU *timing* probe for the formula model.
//!
//! Answers a specific question: on wgpu, is the model *compute* slow, or are we
//! losing wall-clock to per-op dispatch/sync overhead between kernels? The formula
//! stage is autoregressive — one decode step is ~150 tiny batch-1 ops — which is the
//! pattern that punishes a dispatch-per-op backend. This test separates the two:
//!
//!   - **encoder**: one big batched Swin forward (compute-heavy, few large matmuls).
//!     If wgpu kernels are genuinely fast, the encoder is where the GPU wins.
//!   - **decode step**: one autoregressive step (tiny ops). If per-step time is a
//!     roughly-fixed floor that does NOT shrink when we go from batch 1 to batch 32,
//!     that floor is dispatch/sync overhead — the "bad in between", not the models.
//!
//! Every timed region ends with a host read-back (`float_to_vec_f32` /
//! `int_to_vec_i64`), which forces the wgpu queue to drain — so the elapsed time is
//! real GPU wall-clock, not just kernels enqueued lazily.
//!
//! ```text
//! MINERU_FORMULA_DIR=/path/unimernet_hf_small_2503 \
//!   cargo test -p mineru-formula --release --features gpu \
//!     --test gpu_profile -- --ignored --nocapture
//! ```

#![cfg(feature = "gpu")]

use std::path::Path;
use std::time::Instant;

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

/// Median of a small sample (odd/even both fine — we just want a robust center).
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[test]
#[ignore = "needs the gpu feature + a GPU + the checkpoint dir"]
fn gpu_vs_cpu_op_timing() {
    use burn::tensor::{Tensor, TensorData};
    use mineru_burn_common::backend::{cpu_device, gpu_available, gpu_device, Cpu, Gpu};
    use mineru_burn_common::weights::Coverage;
    use mineru_burn_common::{float_to_vec_f32, int_to_vec_i64};
    use mineru_formula::FormulaRecognizer;

    let Ok(dir) = std::env::var("MINERU_FORMULA_DIR") else {
        eprintln!("set MINERU_FORMULA_DIR to the unimernet_hf_small_2503 dir");
        return;
    };
    if !gpu_available() {
        eprintln!("no usable GPU; skipping");
        return;
    }

    let rec_cpu =
        FormulaRecognizer::<Cpu>::from_pretrained(&dir, Coverage::Strict).expect("cpu load");
    let rec_gpu =
        FormulaRecognizer::<Gpu>::from_pretrained_on(&dir, Coverage::Strict, gpu_device())
            .expect("gpu load");

    let ref_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("reference");
    let (px, ps) = read_ref(&ref_dir, "input");
    let make_cpu = || -> Tensor<Cpu, 4> {
        Tensor::from_data(TensorData::new(px.clone(), [ps[0], ps[1], ps[2], ps[3]]), &cpu_device())
    };
    let make_gpu = || -> Tensor<Gpu, 4> {
        Tensor::from_data(TensorData::new(px.clone(), [ps[0], ps[1], ps[2], ps[3]]), &gpu_device())
    };

    let bos = 0u32;

    // ------------------------------------------------------------------
    // 1) ENCODER — one big batched Swin forward. Sync via a host read-back
    //    of the final grid. Warm up once (first wgpu call compiles kernels).
    // ------------------------------------------------------------------
    let time_enc_cpu = || {
        let t = Instant::now();
        let (_e, stages) = rec_cpu.encode_stages(make_cpu());
        let _ = float_to_vec_f32(stages.last().unwrap().clone()); // sync
        t.elapsed().as_secs_f64() * 1e3
    };
    let time_enc_gpu = || {
        let t = Instant::now();
        let (_e, stages) = rec_gpu.encode_stages(make_gpu());
        let _ = float_to_vec_f32(stages.last().unwrap().clone()); // sync (drains queue)
        t.elapsed().as_secs_f64() * 1e3
    };

    let _ = time_enc_cpu();
    let _ = time_enc_gpu(); // warm up kernel compilation

    let enc_cpu_ms = median((0..5).map(|_| time_enc_cpu()).collect());
    let enc_gpu_ms = median((0..5).map(|_| time_enc_gpu()).collect());

    // Keep one real encoder grid per backend to feed the decoder.
    let (_e, stages_cpu) = rec_cpu.encode_stages(make_cpu());
    let (_e, stages_gpu) = rec_gpu.encode_stages(make_gpu());
    let enc_cpu = stages_cpu.last().unwrap().clone();
    let enc_gpu = stages_gpu.last().unwrap().clone();

    // ------------------------------------------------------------------
    // 2) DECODE STEP — the autoregressive quantum. `decoder_step_logits`
    //    re-runs the prefix (non-cached), so timing it at prefix length T
    //    reveals whether cost is a fixed floor (dispatch/sync bound) or
    //    scales with T (compute bound). We time T = 1 and T = 32.
    //    Each call ends with an argmax read-back → queue drains.
    // ------------------------------------------------------------------
    let step_cpu = |ids: &[u32]| {
        let t = Instant::now();
        let logits = rec_cpu.decoder_step_logits(ids, enc_cpu.clone());
        let _ = int_to_vec_i64(logits.argmax(1)); // sync
        t.elapsed().as_secs_f64() * 1e3
    };
    let step_gpu = |ids: &[u32]| {
        let t = Instant::now();
        let logits = rec_gpu.decoder_step_logits(ids, enc_gpu.clone());
        let _ = int_to_vec_i64(logits.argmax(1)); // sync (drains queue)
        t.elapsed().as_secs_f64() * 1e3
    };

    let prefix1 = vec![bos];
    let prefix32: Vec<u32> = std::iter::repeat(bos).take(32).collect();

    let _ = step_cpu(&prefix1);
    let _ = step_gpu(&prefix1); // warm up

    let step1_cpu = median((0..7).map(|_| step_cpu(&prefix1)).collect());
    let step1_gpu = median((0..7).map(|_| step_gpu(&prefix1)).collect());
    let step32_cpu = median((0..7).map(|_| step_cpu(&prefix32)).collect());
    let step32_gpu = median((0..7).map(|_| step_gpu(&prefix32)).collect());

    // ------------------------------------------------------------------
    // 3) CACHED decode step — the REAL production per-token path (KV cache,
    //    one new token per step). This is what dominates the 73% formula
    //    stage. Median over a warmed run of steps, at batch 1 and batch 32.
    //    A cached step is a *fixed* amount of work per token (no prefix
    //    recompute), so its GPU time is the cleanest read on the per-step
    //    dispatch/sync floor.
    // ------------------------------------------------------------------
    let cached_cpu_b1 = rec_cpu.cached_step_times_ms(enc_cpu.clone(), 1, 40, bos);
    let cached_gpu_b1 = rec_gpu.cached_step_times_ms(enc_gpu.clone(), 1, 40, bos);
    // Drop the first few steps (warm-up / cache growth transient), take the median.
    let warm = |v: &[f64]| median(v.iter().skip(5).copied().collect());
    let cstep_cpu_b1 = warm(&cached_cpu_b1);
    let cstep_gpu_b1 = warm(&cached_gpu_b1);

    // Batch 32 needs a 32-lane encoder grid; tile the single grid we have.
    let tile32 = |g: &Tensor<Cpu, 3>| -> Tensor<Cpu, 3> {
        let d = g.dims();
        let flat = float_to_vec_f32(g.clone());
        let mut big = Vec::with_capacity(flat.len() * 32);
        for _ in 0..32 {
            big.extend_from_slice(&flat);
        }
        Tensor::from_data(TensorData::new(big, [32, d[1], d[2]]), &cpu_device())
    };
    let tile32g = |g: &Tensor<Gpu, 3>| -> Tensor<Gpu, 3> {
        let d = g.dims();
        let flat = float_to_vec_f32(g.clone());
        let mut big = Vec::with_capacity(flat.len() * 32);
        for _ in 0..32 {
            big.extend_from_slice(&flat);
        }
        Tensor::from_data(TensorData::new(big, [32, d[1], d[2]]), &gpu_device())
    };
    let enc_cpu32 = tile32(&enc_cpu);
    let enc_gpu32 = tile32g(&enc_gpu);
    let cached_cpu_b32 = rec_cpu.cached_step_times_ms(enc_cpu32.clone(), 32, 40, bos);
    let cached_gpu_b32 = rec_gpu.cached_step_times_ms(enc_gpu32.clone(), 32, 40, bos);
    let cstep_cpu_b32 = warm(&cached_cpu_b32);
    let cstep_gpu_b32 = warm(&cached_gpu_b32);

    // ------------------------------------------------------------------
    // 4) ON-DEVICE loop: same cached steps but the argmax token stays on
    //    the GPU and feeds the next step directly, syncing only every 8
    //    steps. If the per-step host read-back was the floor, per-step
    //    cost drops here. Total ms / n_steps = per-step cost.
    // ------------------------------------------------------------------
    let n = 40usize;
    let od_every = 8usize;
    // warm
    let _ = rec_gpu.ondevice_decode_ms(enc_gpu.clone(), 1, 8, bos, od_every);
    let _ = rec_cpu.ondevice_decode_ms(enc_cpu.clone(), 1, 8, bos, od_every);

    let od_cpu_b1 = rec_cpu.ondevice_decode_ms(enc_cpu.clone(), 1, n, bos, od_every) / n as f64;
    let od_gpu_b1 = rec_gpu.ondevice_decode_ms(enc_gpu.clone(), 1, n, bos, od_every) / n as f64;
    let od_cpu_b32 = rec_cpu.ondevice_decode_ms(enc_cpu32, 32, n, bos, od_every) / n as f64;
    let od_gpu_b32 = rec_gpu.ondevice_decode_ms(enc_gpu32, 32, n, bos, od_every) / n as f64;

    // ------------------------------------------------------------------
    // Report
    // ------------------------------------------------------------------
    let ratio = |cpu: f64, gpu: f64| {
        if gpu > 0.0 {
            cpu / gpu
        } else {
            0.0
        }
    };
    println!("\n================ formula op timing: CPU (flex) vs GPU (wgpu) ================");
    println!("(median ms; each timed region ends with a host read-back to drain the GPU queue)\n");
    println!("{:<34} {:>10} {:>10} {:>10}", "region", "CPU ms", "GPU ms", "GPU speedup");
    println!("{:-<66}", "");
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}x",
        "encoder forward (batched, big matmul)",
        enc_cpu_ms,
        enc_gpu_ms,
        ratio(enc_cpu_ms, enc_gpu_ms)
    );
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}x",
        "decode step, prefix T=1 (tiny ops)",
        step1_cpu,
        step1_gpu,
        ratio(step1_cpu, step1_gpu)
    );
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}x",
        "decode step, prefix T=32",
        step32_cpu,
        step32_gpu,
        ratio(step32_cpu, step32_gpu)
    );
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}x",
        "CACHED step, batch 1 (production)",
        cstep_cpu_b1,
        cstep_gpu_b1,
        ratio(cstep_cpu_b1, cstep_gpu_b1)
    );
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}x",
        "CACHED step, batch 32 (production)",
        cstep_cpu_b32,
        cstep_gpu_b32,
        ratio(cstep_cpu_b32, cstep_gpu_b32)
    );
    println!("{:-<66}", "");
    println!("\n--- ON-DEVICE loop (token stays on GPU, sync every 8 steps) ---");
    println!("{:<34} {:>10} {:>10} {:>10}", "region", "CPU ms", "GPU ms", "GPU speedup");
    println!("{:-<66}", "");
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}x",
        "on-device step, batch 1",
        od_cpu_b1,
        od_gpu_b1,
        ratio(od_cpu_b1, od_gpu_b1)
    );
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}x",
        "on-device step, batch 32",
        od_cpu_b32,
        od_gpu_b32,
        ratio(od_cpu_b32, od_gpu_b32)
    );
    println!("{:-<66}", "");
    println!(
        "\nGPU per-step: per-step-sync {:.2} ms -> on-device {:.2} ms  ({:.2}x faster from killing the sync)",
        cstep_gpu_b1, od_gpu_b1, cstep_gpu_b1 / od_gpu_b1.max(1e-9)
    );
    println!(
        "GPU batch-32: per-step-sync {:.2} ms -> on-device {:.2} ms  ({:.2}x)",
        cstep_gpu_b32, od_gpu_b32, cstep_gpu_b32 / od_gpu_b32.max(1e-9)
    );

    // Per-lane throughput: a batch-32 step produces 32 tokens. Divide the step time
    // by 32 to get cost-per-token, and compare to the batch-1 step (1 token). If the
    // batch-32 step is barely slower than batch-1, the extra 31 lanes rode for ~free
    // on top of a fixed per-step floor — the signature of an overhead-bound step.
    println!("\ncost per TOKEN (cached step / batch):");
    println!(
        "  CPU: b1 {:.2} ms/tok -> b32 {:.2} ms/tok   ({:.1}x cheaper batched)",
        cstep_cpu_b1,
        cstep_cpu_b32 / 32.0,
        cstep_cpu_b1 / (cstep_cpu_b32 / 32.0)
    );
    println!(
        "  GPU: b1 {:.2} ms/tok -> b32 {:.2} ms/tok   ({:.1}x cheaper batched)",
        cstep_gpu_b1,
        cstep_gpu_b32 / 32.0,
        cstep_gpu_b1 / (cstep_gpu_b32 / 32.0)
    );
    println!(
        "  GPU fixed floor: a batch-32 step ({:.1} ms) is only {:.2}x a batch-1 step ({:.1} ms)",
        cstep_gpu_b32,
        cstep_gpu_b32 / cstep_gpu_b1.max(1e-9),
        cstep_gpu_b1
    );

    // The diagnostic: how much of a T=32 step is *fixed* (dispatch/sync floor)?
    // Non-cached decode at T does ~T× the self-attention/FFN work of T=1, so if the
    // step barely grows from T=1 to T=32 the time is dominated by the per-step floor.
    let gpu_growth = if step1_gpu > 0.0 { step32_gpu / step1_gpu } else { 0.0 };
    let cpu_growth = if step1_cpu > 0.0 { step32_cpu / step1_cpu } else { 0.0 };
    println!("\nT=1 -> T=32 step growth (32x more decoder work):");
    println!("  CPU: {cpu_growth:.2}x    GPU: {gpu_growth:.2}x");
    println!("  (growth << 32x means per-step time is a FIXED floor, i.e. overhead-bound,");
    println!("   NOT compute-bound. A large GPU floor = dispatch/sync between tiny ops.)\n");

    // This test never fails on timing (numbers vary by machine); it's a probe.
    // A trivial assert keeps it a real test rather than a silently-skipped one.
    assert!(enc_gpu_ms >= 0.0 && step1_gpu >= 0.0);
}
