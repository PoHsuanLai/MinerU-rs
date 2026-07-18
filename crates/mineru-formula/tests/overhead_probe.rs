//! Diagnostic probe: why is a batch-1 decode step ~48 ms when its arithmetic is
//! worth ~12 ms?
//!
//! **The answer: our matmul kernel is ~9x slower than PyTorch's at this shape.**
//! Not the hardware, not dispatch, not the shape workarounds.
//!
//! The decode is *shaped* like a memory-bound problem — a step streams ~418 MB of
//! decoder weights (8 layers x 33 MB + a 154 MB `lm_head`, fp32) to emit one
//! token, and at batch 1 each byte is used once, so nothing hides the load
//! latency. That framing explains why batching and flex won and why `apple-amx`
//! did nothing. But it does **not** explain the absolute cost, and it should not
//! be mistaken for a wall:
//!
//! | | `[1,768]@[768,768]` | streaming 380 MB of distinct weights |
//! |---|---|---|
//! | torch | 4.15 us (286 Gflop/s) | 2.59 ms (147 GB/s) |
//! | ours | ~37 us (26 Gflop/s) | 48.35 ms (8.6 GB/s) |
//!
//! Torch reaches ~54% of the M4 Pro's ~273 GB/s on the same weight-streaming
//! problem; we reach ~3%. It does it **single-threaded** (1-thread and 8-thread
//! are both 4.1 us) while we burn ~2.8 cores. So neither DRAM bandwidth nor
//! threading is the limit — the per-core kernel is.
//!
//! Hypotheses these probes killed, which is why they are kept:
//!
//! - **Per-op dispatch overhead.** [`per_op_floor`]: a `[1,1]` add costs 1.24 us,
//!   so a ~150-op step pays ~0.2 ms — 0.4% of the step, not the missing 75%.
//! - **Harness contamination.** `clone` is 0.03 us and `into_data` 0.19 us; the
//!   37 us is the kernel itself, not the measurement around it.
//! - **gemm is broken.** It beats a naive scalar loop 7.55x, so it is genuinely
//!   vectorizing — just not competitively at skinny shapes.
//!
//! Beware: these numbers carry real variance (the same matmul measured 91.67 us
//! on a contended run and 37 us clean). Re-measure before trusting a delta
//! smaller than ~2x.
//!
//! ```text
//! cargo test -p mineru-formula --release --test overhead_probe -- --ignored --nocapture
//! ```

use burn::tensor::{Distribution, Tensor};
use mineru_burn_common::backend::{cpu_device, Cpu};
use std::time::Instant;

fn bench<F, const D: usize>(iters: usize, mut f: F) -> f64
where
    F: FnMut() -> Tensor<Cpu, D>,
{
    let warm = f();
    let _ = warm.into_data();
    let t = Instant::now();
    for _ in 0..iters {
        let out = f();
        let _ = out.into_data();
    }
    t.elapsed().as_secs_f64() * 1e6 / iters as f64 // microseconds
}

/// Times the same op class across sizes spanning four orders of magnitude of work.
///
/// `us per Mflop` collapsing as `n` grows (412 -> 30) is the tell: small matmuls
/// pay a cost that has nothing to do with their arithmetic.
#[test]
#[ignore = "diagnostic; release-only, prints a table"]
fn cost_does_not_track_arithmetic() {
    let dev = cpu_device();

    println!("\n=== matmul: [1,n] @ [n,n] — does cost track work or op count? ===");
    println!("{:>6}  {:>10}  {:>14}  {:>12}", "n", "time(us)", "flops", "us per Mflop");
    for n in [64usize, 128, 256, 512, 768, 1024, 2048] {
        let a: Tensor<Cpu, 2> = Tensor::random([1, n], Distribution::Default, &dev);
        let b: Tensor<Cpu, 2> = Tensor::random([n, n], Distribution::Default, &dev);
        let us = bench(200, || a.clone().matmul(b.clone()));
        // A [1,n]@[n,n] matmul is 2*n*n flops.
        let mflop = 2.0 * (n as f64) * (n as f64) / 1e6;
        println!("{n:>6}  {us:>10.2}  {:>14.2}M  {:>12.2}", mflop, us / mflop);
    }

    println!("\n=== elementwise add: [1,n] + [1,n] ===");
    println!("{:>6}  {:>10}  {:>14}", "n", "time(us)", "us per Kelem");
    for n in [64usize, 768, 4096, 65536, 1_048_576] {
        let a: Tensor<Cpu, 2> = Tensor::random([1, n], Distribution::Default, &dev);
        let b: Tensor<Cpu, 2> = Tensor::random([1, n], Distribution::Default, &dev);
        let us = bench(200, || a.clone() + b.clone());
        println!("{n:>6}  {us:>10.2}  {:>14.3}", us / (n as f64 / 1000.0));
    }

    println!("\nIf `us per Mflop` collapses as n grows, small ops are paying a fixed");
    println!("cost that has nothing to do with arithmetic — that is the decode's 75%.");
}

/// Isolates the floor: how long does the cheapest possible tensor op take?
///
/// A `[1,1]` add does one flop. Whatever it costs is pure overhead, and a decode
/// step dispatches on the order of a hundred ops (8 layers x ~12 ops + head).
#[test]
#[ignore = "diagnostic; release-only, prints a table"]
fn per_op_floor() {
    let dev = cpu_device();
    let n = 2000;

    let one: Tensor<Cpu, 2> = Tensor::random([1, 1], Distribution::Default, &dev);
    let add1 = bench(n, || one.clone() + one.clone());

    let row: Tensor<Cpu, 2> = Tensor::random([1, 768], Distribution::Default, &dev);
    let add768 = bench(n, || row.clone() + row.clone());

    let m: Tensor<Cpu, 2> = Tensor::random([768, 768], Distribution::Default, &dev);
    let mm = bench(200, || row.clone().matmul(m.clone()));

    let sm = bench(n, || burn::tensor::activation::softmax(row.clone(), 1));
    let rs = bench(n, || row.clone().reshape([768, 1]));

    println!("\n=== the per-op floor (batch-1 decode shapes) ===");
    println!("add   [1,1]     (1 flop)        {add1:8.2} us  <- pure overhead");
    println!("add   [1,768]   (768 flops)     {add768:8.2} us");
    println!("matmul[1,768]@[768,768]         {mm:8.2} us  (1.2 Mflop)");
    println!("softmax [1,768]                 {sm:8.2} us");
    println!("reshape [1,768] -> [768,1]      {rs:8.2} us");

    // A decode step is ~8 layers x (4 proj + 2 ffn + 2 scores + 2 ctx + 2 softmax
    // + 2 norms + adds/reshapes) + embed + lm_head: on the order of 120-160 ops.
    for ops in [120.0, 160.0] {
        println!(
            "\n{ops:.0} ops x {add1:.2} us floor = {:.1} ms of pure dispatch per step",
            ops * add1 / 1000.0
        );
    }
    println!("(measured step: ~48 ms; ops account for ~12 ms)");
}

/// Tests the bandwidth-bound hypothesis directly.
///
/// `[batch, 768] @ [768, 768]` reads the same 2.4 MB weight matrix no matter the
/// batch. If the kernel is memory-bound, growing the batch adds arithmetic that
/// rides along on weights already in cache — near-flat wall-clock, rising
/// Gflop/s. If it were compute-bound, time would scale linearly with batch.
#[test]
#[ignore = "diagnostic; release-only, prints a table"]
fn matmul_is_bandwidth_bound() {
    let dev = cpu_device();
    let w: Tensor<Cpu, 2> = Tensor::random([768, 768], Distribution::Default, &dev);

    println!("\n=== [batch,768] @ [768,768] — same weights, more work ===");
    println!("{:>6}  {:>10}  {:>12}  {:>10}", "batch", "time(us)", "Gflop/s", "us/row");
    for b in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let a: Tensor<Cpu, 2> = Tensor::random([b, 768], Distribution::Default, &dev);
        let us = bench(100, || a.clone().matmul(w.clone()));
        let gf = 2.0 * (b as f64) * 768.0 * 768.0 / (us * 1e-6) / 1e9;
        println!("{b:>6}  {us:>10.2}  {gf:>12.1}  {:>10.2}", us / b as f64);
    }
    println!("\nFlat wall-clock + rising Gflop/s = memory-bound: the weights, not the");
    println!("arithmetic, set the cost. That is why batching won 2x and why a faster");
    println!("matmul kernel cannot fix batch-1 decode.");
}
