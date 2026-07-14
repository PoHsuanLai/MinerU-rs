//! Backend-portable host reads of tensor data.
//!
//! Copying a tensor to the host with `into_data().into_vec::<E>()` only succeeds
//! when the *on-device* element type happens to equal `E`. That assumption holds
//! for the CPU (`NdArray`) backend, whose ints are `i64` and floats `f32`, but
//! **not** for `wgpu`, whose [`Gpu`](crate::backend::Gpu) alias stores ints as
//! `i32`. On that backend `into_vec::<i64>()` returns [`DataError::TypeMismatch`]
//! and any caller that swallowed the error into an empty/zeroed vec silently
//! corrupted its result.
//!
//! These helpers read to a fixed host element type *regardless* of the backend's
//! storage dtype by coercing through [`TensorData::convert`], which upcasts
//! `i32 → i64` (or `f16/flex32 → f32`, etc.) before the copy. They are the single
//! correct way for model code to pull an `Int`/float tensor to host `Vec`.

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

/// Reads an `Int` tensor to a host `Vec<i64>`, coercing the backend's storage
/// dtype (e.g. `wgpu`'s `i32`) to `i64` first.
///
/// Row-major over the tensor's dims. Backend-portable: unlike
/// `into_data().into_vec::<i64>()` this never fails on a dtype mismatch.
pub fn int_to_vec_i64<B: Backend, const D: usize>(x: Tensor<B, D, Int>) -> Vec<i64> {
    x.into_data().convert::<i64>().into_vec::<i64>().unwrap_or_default()
}

/// Reads a float tensor to a host `Vec<f32>`, coercing the backend's storage
/// dtype (e.g. `f16`/`flex32`) to `f32` first.
///
/// Row-major over the tensor's dims. Backend-portable: unlike
/// `into_data().into_vec::<f32>()` this never fails on a dtype mismatch.
pub fn float_to_vec_f32<B: Backend, const D: usize>(x: Tensor<B, D>) -> Vec<f32> {
    x.into_data().convert::<f32>().into_vec::<f32>().unwrap_or_default()
}
