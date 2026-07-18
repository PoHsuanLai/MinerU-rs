//! Diagnostic probe: is the slow batch-1 matmul the kernel, or the code around it?
//!
//! PyTorch does `[1,768] @ [768,768]` (fp32, CPU, the shape our decode hammers) in
//! **4.15 us**; the same op through Burn takes **~37 us**. This probe pins that
//! ratio down to the kernel by ruling out everything else:
//!
//! - **The harness is free.** `clone` of the `[768,768]` weight is 0.03 us and
//!   `into_data` of the result 0.19 us, against a 37 us matmul. An earlier probe
//!   reported 91.67 us for the same expression on a contended run — treat any
//!   delta under ~2x in these numbers as noise.
//! - **gemm is not broken.** It beats a naive scalar triple-loop by ~7.5x, so it
//!   really is vectorizing. It just does not reach torch at skinny shapes.
//! - **Threading is not the lever.** Torch hits 286 Gflop/s *single-threaded*
//!   (1-thread 4.12 us vs 8-thread 4.13 us) while our decode burns ~2.8 cores to
//!   reach 26 Gflop/s.
//!
//! ```text
//! cargo test -p mineru-formula --release --test gemm_direct -- --ignored --nocapture
//! ```

use burn::tensor::{Distribution, Tensor};
use mineru_burn_common::backend::{cpu_device, Cpu};
use std::time::Instant;

/// Torch's measured cost for the same op on the same machine (fp32, CPU).
const TORCH_US: f64 = 4.15;

#[test]
#[ignore = "diagnostic; release-only"]
fn burn_matmul_vs_naive_and_torch() {
    let dev = cpu_device();
    let n = 500;
    let a: Tensor<Cpu, 2> = Tensor::random([1, 768], Distribution::Default, &dev);
    let w: Tensor<Cpu, 2> = Tensor::random([768, 768], Distribution::Default, &dev);

    // The matmul alone: no host read, so only the kernel is timed.
    for _ in 0..20 {
        std::hint::black_box(a.clone().matmul(w.clone()));
    }
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(a.clone().matmul(w.clone()));
    }
    let burn_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;

    // The harness, priced separately: if these are not ~0, the number above is not
    // the kernel.
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(w.clone());
    }
    let clone_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(a.clone().into_data());
    }
    let read_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;

    // A naive scalar loop: the floor a library must beat to justify itself.
    let av: Vec<f32> = (0..768).map(|i| (i as f32).sin()).collect();
    let wv: Vec<f32> = (0..768 * 768).map(|i| (i as f32).cos()).collect();
    let mut out = vec![0.0f32; 768];
    let t = Instant::now();
    for _ in 0..n {
        for (j, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (k, av_k) in av.iter().enumerate() {
                acc += av_k * wv.get(k * 768 + j).copied().unwrap_or_default();
            }
            *o = acc;
        }
        std::hint::black_box(&out);
    }
    let naive_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;

    let gflops = |us: f64| 2.0 * 768.0 * 768.0 / (us * 1e-6) / 1e9;
    println!("\n=== [1,768] @ [768,768], fp32 ===");
    println!("torch (measured)   {TORCH_US:8.2} us  ({:6.1} Gflop/s)", gflops(TORCH_US));
    println!("burn matmul        {burn_us:8.2} us  ({:6.1} Gflop/s)", gflops(burn_us));
    println!("naive scalar loop  {naive_us:8.2} us  ({:6.1} Gflop/s)", gflops(naive_us));
    println!("\nharness, for scale:");
    println!("  clone [768,768]  {clone_us:8.2} us");
    println!("  into_data [1,768]{read_us:8.2} us");
    println!("\nburn beats naive by {:.1}x, but torch beats burn by {:.1}x",
        naive_us / burn_us, burn_us / TORCH_US);
}
