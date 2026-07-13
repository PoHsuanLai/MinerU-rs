//! Prompt strings and sampling parameters for the MinerU2.5 VLM.
//!
//! Values mirror `mineru-vl-utils`' `DEFAULT_PROMPTS` and `MinerUSamplingParams`.
//! Every prompt begins with a literal newline, which is kept.

/// System prompt sent with every request.
pub const SYSTEM: &str = "You are a helpful assistant.";

/// Step-1 prompt: detect the page layout.
pub const LAYOUT: &str = "\nLayout Detection:";

/// The step-2 prompt for a block, chosen by its layout label.
pub fn extraction_prompt(label: &str) -> &'static str {
    match label {
        "table" => "\nTable Recognition:",
        "equation" => "\nFormula Recognition:",
        "image" | "chart" => "\nImage Analysis:",
        _ => "\nText Recognition:",
    }
}

/// Sampling parameters for one request. Non-standard fields are sent via the
/// request's `extra_body` since they are vLLM/SGLang extensions.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: i32,
}

impl Default for Sampling {
    fn default() -> Self {
        // The layout step and the base configuration use these values.
        Self {
            temperature: 0.0,
            top_p: 0.01,
            top_k: 1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 100,
        }
    }
}

impl Sampling {
    /// Per-label overrides for the step-2 extraction pass.
    pub fn for_extraction(label: &str) -> Self {
        let mut s = Self::default();
        match label {
            "table" => {
                s.presence_penalty = 1.0;
                s.frequency_penalty = 0.005;
            }
            "text" | "equation" | "image" | "chart" => {
                s.presence_penalty = 1.0;
                s.frequency_penalty = 0.05;
            }
            _ => {
                s.presence_penalty = 1.0;
                s.frequency_penalty = 0.05;
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_have_leading_newline() {
        assert!(LAYOUT.starts_with('\n'));
        assert!(extraction_prompt("table").starts_with('\n'));
    }

    #[test]
    fn table_sampling_uses_low_frequency_penalty() {
        assert_eq!(Sampling::for_extraction("table").frequency_penalty, 0.005);
        assert_eq!(Sampling::for_extraction("text").frequency_penalty, 0.05);
    }

    #[test]
    fn prompt_selection_by_label() {
        assert_eq!(extraction_prompt("equation"), "\nFormula Recognition:");
        assert_eq!(extraction_prompt("paragraph"), "\nText Recognition:");
    }
}
