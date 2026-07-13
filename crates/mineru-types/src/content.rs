//! Content newtypes.
//!
//! These wrap `String`/`f32` so the type system can tell apart values that would
//! otherwise all be interchangeable strings — a raw HTML fragment cannot be passed
//! where LaTeX or plain markdown text is expected.

use serde::{Deserialize, Serialize};

/// A confidence score in `0.0..=1.0` emitted by a model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Score(pub f32);

/// An HTML fragment (e.g. a recognized table's markup).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Html(pub String);

/// A LaTeX fragment (e.g. a recognized formula).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Latex(pub String);

/// An OCR / document language tag (e.g. `"ch"`, `"en"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lang(pub String);

/// A reference to an extracted raster image, stored as a path relative to the
/// output's image directory. Resolved against a base directory at render time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRef(pub String);

macro_rules! impl_str_newtype {
    ($($ty:ident),+ $(,)?) => {$(
        impl $ty {
            /// Borrows the inner string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl From<String> for $ty {
            fn from(s: String) -> Self {
                Self(s)
            }
        }
        impl From<&str> for $ty {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }
    )+};
}

impl_str_newtype!(Html, Latex, Lang, ImageRef);
