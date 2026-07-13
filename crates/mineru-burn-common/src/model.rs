//! A tiny uniform inference trait.
//!
//! Every model crate produces something callable as "input in, output out". This
//! trait pins that shape so higher layers can treat models uniformly, while each
//! crate keeps concrete, strongly-typed `Input`/`Output` associated types.

use crate::error::Result;

/// A model that maps an input to an output, fallibly.
///
/// Implementors choose concrete `Input`/`Output` types (e.g. an `RgbImage` in, a
/// list of detected boxes out). Keeping this minimal avoids leaking any single
/// model's shape into the shared harness.
pub trait Model {
    /// The model's input type (e.g. a preprocessed image or tensor).
    type Input;
    /// The model's output type (e.g. decoded text, boxes, or a class map).
    type Output;

    /// Runs inference on `input`.
    fn predict(&self, input: Self::Input) -> Result<Self::Output>;
}
