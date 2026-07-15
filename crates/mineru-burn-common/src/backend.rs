//! Backend type aliases and device access.
//!
//! Every model crate should refer to backends through these aliases rather than
//! naming concrete Burn backend types, so that the choice of CPU vs GPU stays in
//! one place. The CPU backend ([`Cpu`]) is always available; the GPU backend
//! ([`Gpu`]) is compiled only when the `gpu` feature is enabled.

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;

/// The default CPU backend: `ndarray` with `f32` elements.
///
/// This backend requires no GPU toolchain and is used for development, tests, and
/// as the fallback everywhere. Model crates should default to `Cpu` and only opt
/// into [`Gpu`] behind their own feature flag.
pub type Cpu = NdArray<f32>;

/// Returns the default device for the [`Cpu`] backend.
///
/// `NdArray` is single-device, so this is always the CPU. Kept as a function
/// (rather than a `const`) to mirror the GPU case, where device selection is real.
pub fn cpu_device() -> NdArrayDevice {
    NdArrayDevice::default()
}

#[cfg(feature = "gpu")]
mod gpu {
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::Tensor;

    /// The optional GPU backend: `wgpu` with `f32` elements.
    ///
    /// Only compiled when the `gpu` feature is enabled. Never required for CPU
    /// inference or the test suite.
    pub type Gpu = Wgpu<f32, i32>;

    /// Returns the default device for the [`Gpu`] backend.
    pub fn gpu_device() -> WgpuDevice {
        WgpuDevice::default()
    }

    /// Probes whether a usable wgpu GPU is actually available.
    ///
    /// Returns `true` only if the default device can be initialized *and* a
    /// trivial tensor op runs on it end-to-end (allocate → compute → read back).
    /// A machine with no Metal/Vulkan adapter, broken drivers, or a headless
    /// environment fails one of those steps; the probe reports `false` so callers
    /// can fall back to CPU instead of committing to a device that would panic
    /// mid-pipeline.
    ///
    /// The wgpu init path can `panic!` (rather than return an error) when no
    /// adapter is present, so the probe runs inside [`std::panic::catch_unwind`]
    /// and treats a panic as "unavailable". It is a one-off, cheap check meant to
    /// be called once before loading models onto the GPU.
    #[must_use]
    pub fn gpu_available() -> bool {
        // The probe both allocates a device buffer and reads it back, so a broken
        // adapter that constructs but cannot execute is still caught. `catch_unwind`
        // guards the wgpu init panic; the closure returns the checked value.
        std::panic::catch_unwind(|| {
            let device = gpu_device();
            let a = Tensor::<Gpu, 1>::from_data([1.0f32, 2.0, 3.0], &device);
            let sum = a.sum().into_scalar();
            // 1 + 2 + 3 = 6; a finite, correct read-back confirms the device works.
            (sum - 6.0).abs() < 1e-3
        })
        .unwrap_or(false)
    }
}

#[cfg(feature = "gpu")]
pub use gpu::{Gpu, gpu_available, gpu_device};
