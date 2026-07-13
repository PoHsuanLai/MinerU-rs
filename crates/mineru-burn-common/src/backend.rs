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

    /// The optional GPU backend: `wgpu` with `f32` elements.
    ///
    /// Only compiled when the `gpu` feature is enabled. Never required for CPU
    /// inference or the test suite.
    pub type Gpu = Wgpu<f32, i32>;

    /// Returns the default device for the [`Gpu`] backend.
    pub fn gpu_device() -> WgpuDevice {
        WgpuDevice::default()
    }
}

#[cfg(feature = "gpu")]
pub use gpu::{Gpu, gpu_device};
