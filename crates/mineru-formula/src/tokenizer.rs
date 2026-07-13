//! LaTeX tokenizer: loads the checkpoint's HuggingFace `tokenizer.json` and
//! decodes token ids back to a LaTeX string.
//!
//! The UniMerNet checkpoint ships a standard HF fast-tokenizer (`tokenizer.json`),
//! so we defer to the [`tokenizers`] crate rather than reimplementing BPE. Only
//! **decoding** is needed for inference (id sequence -> string); the entry point
//! never tokenizes text.
//!
//! Python reference: `TokenizerWrapper.token2str` in `modeling_unimernet.py` calls
//! `batch_decode(..., skip_special_tokens=True)` and then `ftfy.fix_text`. We skip
//! special tokens the same way. We do **not** run `ftfy` (mojibake repair) — the
//! model output is already clean LaTeX in practice; this is a noted, minor
//! deviation. The heavier LaTeX whitespace cleanup is [`crate::latex_cleanup`].

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

/// A thin wrapper over an HF [`Tokenizer`] exposing decode + the special ids.
pub struct LatexTokenizer {
    inner: Tokenizer,
    /// Padding token id, read from the tokenizer where available.
    pub pad_token_id: Option<u32>,
    /// BOS / decoder-start token id.
    pub bos_token_id: Option<u32>,
    /// EOS token id.
    pub eos_token_id: Option<u32>,
}

impl LatexTokenizer {
    /// Loads a tokenizer from a `tokenizer.json` file path.
    ///
    /// # Errors
    /// Returns [`Error::Tokenizer`] if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("failed to load tokenizer: {e}")))?;
        Ok(Self::from_tokenizer(inner))
    }

    /// Wraps an already-constructed [`Tokenizer`], resolving special-token ids by
    /// their conventional surface forms (`<s>`, `</s>`, `<pad>`).
    pub fn from_tokenizer(inner: Tokenizer) -> Self {
        let id = |t: &str| inner.token_to_id(t);
        Self {
            pad_token_id: id("<pad>"),
            bos_token_id: id("<s>"),
            eos_token_id: id("</s>"),
            inner,
        }
    }

    /// Vocabulary size (with added tokens), matching `len(tokenizer)`.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Decodes a token-id sequence to a LaTeX string, skipping special tokens
    /// (`skip_special_tokens=True`, matching the Python).
    ///
    /// # Errors
    /// Returns [`Error::Tokenizer`] if the underlying decode fails.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(format!("decode failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::Tokenizer;

    /// Builds a tiny word-level tokenizer with a known vocab for decode tests,
    /// avoiding any dependency on the multi-hundred-MB checkpoint.
    fn tiny_tokenizer() -> LatexTokenizer {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("<s>".to_string(), 0u32);
        vocab.insert("<pad>".to_string(), 1u32);
        vocab.insert("</s>".to_string(), 2u32);
        vocab.insert("\\frac".to_string(), 3u32);
        vocab.insert("{a}".to_string(), 4u32);
        vocab.insert("{b}".to_string(), 5u32);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<pad>".to_string())
            .build()
            .expect("build word-level model");
        let mut tok = Tokenizer::new(model);
        // Mark specials so skip_special_tokens works.
        tok.add_special_tokens(&[
            tokenizers::AddedToken::from("<s>", true),
            tokenizers::AddedToken::from("<pad>", true),
            tokenizers::AddedToken::from("</s>", true),
        ]);
        LatexTokenizer::from_tokenizer(tok)
    }

    #[test]
    fn special_ids_resolved() {
        let t = tiny_tokenizer();
        assert_eq!(t.bos_token_id, Some(0));
        assert_eq!(t.pad_token_id, Some(1));
        assert_eq!(t.eos_token_id, Some(2));
    }

    #[test]
    fn decode_skips_special_tokens() {
        let t = tiny_tokenizer();
        // <s> \frac {a} {b} </s>  ->  should drop <s> and </s>.
        let out = t.decode(&[0, 3, 4, 5, 2]).expect("decode");
        assert!(out.contains("\\frac"));
        assert!(!out.contains("<s>"));
        assert!(!out.contains("</s>"));
    }

    #[test]
    fn decode_known_ids() {
        let t = tiny_tokenizer();
        let out = t.decode(&[3, 4]).expect("decode");
        assert!(out.contains("\\frac"));
        assert!(out.contains("{a}"));
    }
}
