//! Diagnostic probe: inside one decode step, what costs what?
//!
//! Not a correctness gate — a measurement. The decode is ~67% of pipeline
//! wall-clock and Python does the same work ~5x faster while using *less than one
//! core* (83% CPU vs our 276%), so the gap is per-core kernel efficiency, not
//! threading. This times each op class at the real batch-1 decode shapes to say
//! where a step's time actually goes: the projections (pure gemm), the attention
//! matmuls, the softmax, or the `swap_dims`/`reshape` data movement around them.
//!
//! The `attend` path writes `probs @ v` as `(vᵀ @ probsᵀ)ᵀ` to dodge a wgpu
//! stride bug, with a comment claiming it is a "no-op on CPU". That claim is an
//! assumption; `transposed_matmul_vs_plain` measures it — and it holds (+0.1% /
//! -4.4%). The workaround is free; it was a prime suspect and is innocent.
//!
//! **What this probe establishes: the ops are not the problem.** They sum to
//! ~12 ms against a ~48 ms real step. See `overhead_probe` for where the other
//! 75% goes (weight traffic, not arithmetic).
//!
//! ```text
//! cargo test -p mineru-formula --release --test op_profile -- --ignored --nocapture
//! ```

use burn::tensor::{activation::softmax, Tensor};
use mineru_burn_common::backend::{cpu_device, Cpu};
use std::time::Instant;

/// Real UniMerNet-small decoder shapes (see `MBartConfig::default`).
const D_MODEL: usize = 768;
const FFN: usize = 3072;
const HEADS: usize = 16;
const SQUEEZE_DIM: usize = D_MODEL / 2; // qk_squeeze = 2
const HEAD_DIM: usize = D_MODEL / HEADS; // 48
const QK_HEAD_DIM: usize = SQUEEZE_DIM / HEADS; // 24
const LAYERS: usize = 8;
const VOCAB: usize = 50000;
/// Encoder grid for a 564x120 crop, and a mid-decode cache length.
const SRC: usize = 126;
const CACHE: usize = 40;

/// Times `f` over `iters` runs after a warmup, returning mean milliseconds.
///
/// `f` must return a tensor; we force a host read so lazy/async work cannot be
/// timed as free. Burn's CPU backend is eager, but reading keeps this honest if
/// the probe is ever pointed at a lazy backend.
fn bench<F, const D: usize>(iters: usize, mut f: F) -> f64
where
    F: FnMut() -> Tensor<Cpu, D>,
{
    let sink = f();
    let _ = sink.into_data();
    let t = Instant::now();
    for _ in 0..iters {
        let out = f();
        let _ = out.into_data();
    }
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[test]
#[ignore = "diagnostic; release-only, prints a table"]
fn decode_step_op_costs() {
    let dev = cpu_device();
    let n = 200;

    println!("\n=== one decode step, batch 1, real shapes ===");
    println!("d_model={D_MODEL} ffn={FFN} heads={HEADS} src={SRC} cache={CACHE} layers={LAYERS}\n");

    // ---- Projections: [1, 1, 768] @ [768, 768]. Four per layer (q/k/v/out).
    let x: Tensor<Cpu, 3> = Tensor::random([1, 1, D_MODEL], burn::tensor::Distribution::Default, &dev);
    let w: Tensor<Cpu, 2> = Tensor::random([D_MODEL, D_MODEL], burn::tensor::Distribution::Default, &dev);
    let proj = bench(n, || x.clone().matmul(w.clone().unsqueeze()));
    println!("proj    [1,1,768]@[768,768]      {proj:8.4} ms   (x4/layer)");

    // ---- FFN: the widest matmuls in the step.
    let w1: Tensor<Cpu, 2> = Tensor::random([D_MODEL, FFN], burn::tensor::Distribution::Default, &dev);
    let w2: Tensor<Cpu, 2> = Tensor::random([FFN, D_MODEL], burn::tensor::Distribution::Default, &dev);
    let ffn_up = bench(n, || x.clone().matmul(w1.clone().unsqueeze()));
    let h: Tensor<Cpu, 3> = Tensor::random([1, 1, FFN], burn::tensor::Distribution::Default, &dev);
    let ffn_dn = bench(n, || h.clone().matmul(w2.clone().unsqueeze()));
    println!("ffn_up  [1,1,768]@[768,3072]     {ffn_up:8.4} ms   (x1/layer)");
    println!("ffn_dn  [1,1,3072]@[3072,768]    {ffn_dn:8.4} ms   (x1/layer)");

    // ---- Self-attention scores over the KV cache.
    let q: Tensor<Cpu, 4> = Tensor::random([1, HEADS, 1, QK_HEAD_DIM], burn::tensor::Distribution::Default, &dev);
    let k: Tensor<Cpu, 4> = Tensor::random([1, HEADS, CACHE, QK_HEAD_DIM], burn::tensor::Distribution::Default, &dev);
    let scores_self = bench(n, || q.clone().matmul(k.clone().swap_dims(2, 3)));
    println!("attn_scores(self, cache={CACHE})    {scores_self:8.4} ms   (x1/layer)");

    // ---- Cross-attention scores over the encoder grid.
    let kx: Tensor<Cpu, 4> = Tensor::random([1, HEADS, SRC, QK_HEAD_DIM], burn::tensor::Distribution::Default, &dev);
    let scores_cross = bench(n, || q.clone().matmul(kx.clone().swap_dims(2, 3)));
    println!("attn_scores(cross, src={SRC})     {scores_cross:8.4} ms   (x1/layer)");

    // ---- Softmax over the score row.
    let s: Tensor<Cpu, 4> = Tensor::random([1, HEADS, 1, SRC], burn::tensor::Distribution::Default, &dev);
    let sm = bench(n, || softmax(s.clone(), 3));
    println!("softmax [1,16,1,126]             {sm:8.4} ms   (x2/layer)");

    // ---- The lm_head: the single widest matmul in a step.
    let wv: Tensor<Cpu, 2> = Tensor::random([D_MODEL, VOCAB], burn::tensor::Distribution::Default, &dev);
    let head = bench(50, || x.clone().matmul(wv.clone().unsqueeze()));
    println!("lm_head [1,1,768]@[768,50000]    {head:8.4} ms   (x1/step)");

    // ---- Data movement: what the shape juggling costs on its own.
    let ctx: Tensor<Cpu, 4> = Tensor::random([1, HEADS, 1, HEAD_DIM], burn::tensor::Distribution::Default, &dev);
    let swap = bench(n, || ctx.clone().swap_dims(1, 2));
    let swap_reshape = bench(n, || ctx.clone().swap_dims(1, 2).reshape([1, 1, D_MODEL]));
    println!("swap_dims(1,2) [1,16,1,48]       {swap:8.4} ms");
    println!("  + reshape -> [1,1,768]         {swap_reshape:8.4} ms   (x2/layer)");

    // A step is: 8 layers x (4 self proj + 4 cross proj + 2 ffn + 2 scores + 2 ctx
    // matmuls + 2 softmax + shape work), then one lm_head.
    let per_layer = 8.0 * proj + ffn_up + ffn_dn + scores_self + scores_cross + 2.0 * sm;
    let est = LAYERS as f64 * per_layer + head;
    println!("\n--- rough step estimate ---");
    println!("per layer (8 proj + ffn + attn)  {per_layer:8.4} ms");
    println!("x{LAYERS} layers + lm_head           {est:8.4} ms");
    println!("measured decode is ~59 ms/token; this accounts for {:.0}%", est / 59.0 * 100.0);
}

/// Measures the `(vᵀ @ probsᵀ)ᵀ` workaround against the plain `probs @ v`.
///
/// `attend` claims the rewrite is a "no-op on CPU". If that is true these are
/// equal; if the transposes force a materializing copy, it is a real per-layer
/// tax paid on every decode step to work around a *wgpu* bug.
#[test]
#[ignore = "diagnostic; release-only, prints a table"]
fn transposed_matmul_vs_plain() {
    let dev = cpu_device();
    let n = 300;

    println!("\n=== `probs @ v` vs the wgpu-workaround `(vT @ probsT)T` ===");
    for &src in &[CACHE, SRC] {
        let probs: Tensor<Cpu, 4> =
            Tensor::random([1, HEADS, 1, src], burn::tensor::Distribution::Default, &dev);
        // `v` as `shape_v` leaves it: a batch-permuted view.
        let v: Tensor<Cpu, 4> =
            Tensor::random([1, src, HEADS, HEAD_DIM], burn::tensor::Distribution::Default, &dev)
                .swap_dims(1, 2);

        let plain = bench(n, || probs.clone().matmul(v.clone()));
        let workaround = bench(n, || {
            v.clone().swap_dims(2, 3).matmul(probs.clone().swap_dims(2, 3)).swap_dims(2, 3)
        });
        let delta = (workaround - plain) / plain * 100.0;
        println!(
            "src={src:4}: plain {plain:7.4} ms | workaround {workaround:7.4} ms | {delta:+6.1}%"
        );
    }
    println!("\n(x2 per layer x 8 layers x every token — a per-step tax if non-zero)");
}
