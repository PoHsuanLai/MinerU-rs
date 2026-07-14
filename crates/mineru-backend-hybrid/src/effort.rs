//! The hybrid parse-**effort** knob and its behavioral branches.
//!
//! Python reference: `hybrid_analyze.py`
//! - `HYBRID_ANALYZE_EFFORTS = {"medium", "high"}`
//! - `_validate_parse_effort` — rejects anything else with a `ValueError`.
//! - `_resolve_effective_image_analysis` — `medium` forces image-analysis **off**
//!   for the fast path; `high` honors the caller's `image_analysis` flag.
//!
//! The two efforts also select *how the VLM is driven* in the Python:
//! - `medium` → `batch_extract_with_layout`: the VLM extracts each region using
//!   the **pipeline layout** as the external block list (no VLM layout pass).
//! - `high` → `batch_two_step_extract`: the VLM runs its **own** two-step
//!   layout+extract, and the pipeline layout is used only for title-splitting and
//!   OCR-det sidecars.
//!
//! This enum is the Rust-native replacement for the stringly-typed Python effort;
//! [`Effort::validate`] parses the same two strings and the same error, and the
//! per-effort behavior is expressed as methods rather than scattered `if effort ==`
//! branches.

use crate::error::{Error, Result};

/// How hard the hybrid backend works to extract each region.
///
/// The variants correspond one-to-one to the Python `"medium"` / `"high"` effort
/// strings. `Medium` is the fast path (pipeline layout drives the VLM, image
/// analysis forced off); `High` is the thorough path (VLM runs its own layout and
/// extraction, honoring the image-analysis flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Effort {
    /// Fast path: pipeline layout drives per-region VLM extraction; image analysis
    /// is forced off. The Python default.
    #[default]
    Medium,
    /// Thorough path: the VLM runs its own two-step layout+extraction and image
    /// analysis is honored.
    High,
}

impl Effort {
    /// Validates and parses an effort string (`"medium"` or `"high"`).
    ///
    /// Mirrors the Python `_validate_parse_effort`: any other value is an
    /// [`Error::InvalidEffort`] rather than silently picking a strength branch.
    pub fn validate(effort: &str) -> Result<Self> {
        match effort {
            "medium" => Ok(Effort::Medium),
            "high" => Ok(Effort::High),
            other => Err(Error::InvalidEffort(other.to_owned())),
        }
    }

    /// The canonical string for this effort (round-trips [`Effort::validate`]).
    ///
    /// Used to stamp the `_effort` metadata field the Python `init_middle_json`
    /// writes, so downstream consumers see the same value.
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Medium => "medium",
            Effort::High => "high",
        }
    }

    /// Resolves the *effective* image-analysis flag for this effort.
    ///
    /// Mirrors `_resolve_effective_image_analysis`: `medium` forces it **off** to
    /// keep the fast path fast; `high` returns the caller's requested value.
    pub fn effective_image_analysis(self, requested: bool) -> bool {
        match self {
            Effort::Medium => false,
            Effort::High => requested,
        }
    }

    /// Whether the VLM should run its own layout pass (`high`) rather than being
    /// driven by the pipeline layout (`medium`).
    ///
    /// In the Python this is the `batch_two_step_extract` vs
    /// `batch_extract_with_layout` split.
    pub fn vlm_runs_own_layout(self) -> bool {
        matches!(self, Effort::High)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_the_two_efforts() {
        assert_eq!(Effort::validate("medium").unwrap(), Effort::Medium);
        assert_eq!(Effort::validate("high").unwrap(), Effort::High);
    }

    #[test]
    fn validate_rejects_others() {
        let err = Effort::validate("low").unwrap_err();
        match err {
            Error::InvalidEffort(s) => assert_eq!(s, "low"),
            other => panic!("expected InvalidEffort, got {other:?}"),
        }
        assert!(Effort::validate("").is_err());
        assert!(Effort::validate("Medium").is_err(), "case-sensitive like Python");
    }

    #[test]
    fn as_str_roundtrips() {
        for e in [Effort::Medium, Effort::High] {
            assert_eq!(Effort::validate(e.as_str()).unwrap(), e);
        }
    }

    #[test]
    fn medium_forces_image_analysis_off() {
        assert!(!Effort::Medium.effective_image_analysis(true));
        assert!(!Effort::Medium.effective_image_analysis(false));
    }

    #[test]
    fn high_honors_requested_image_analysis() {
        assert!(Effort::High.effective_image_analysis(true));
        assert!(!Effort::High.effective_image_analysis(false));
    }

    #[test]
    fn only_high_runs_own_vlm_layout() {
        assert!(!Effort::Medium.vlm_runs_own_layout());
        assert!(Effort::High.vlm_runs_own_layout());
    }

    #[test]
    fn default_is_medium() {
        assert_eq!(Effort::default(), Effort::Medium);
    }
}
