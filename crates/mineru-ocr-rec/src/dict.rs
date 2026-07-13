//! Character dictionary and CTC label mapping.
//!
//! A faithful port of `BaseRecLabelDecode` / `CTCLabelDecode` construction from
//! `pytorchocr/postprocess/rec_postprocess.py`. The on-disk dictionary is one
//! character per line (e.g. `ppocrv6_dict.txt`). The CTC character list is then:
//!
//! ```text
//! ['blank'] + <file lines> + [' ' if use_space_char]
//! ```
//!
//! so index `0` is the CTC blank and index `i` (for `i >= 1`) is the `(i-1)`-th
//! entry of that list. [`CharDict::decode`] turns a collapsed, blank-free sequence
//! of class indices into a string.

use std::path::Path;

use crate::error::{Error, Result};

/// A CTC character dictionary: the ordered class list with blank at index 0.
#[derive(Debug, Clone)]
pub struct CharDict {
    /// `characters[i]` is the label for class index `i`. `characters[0]` is the
    /// blank placeholder and is never emitted.
    characters: Vec<String>,
}

impl CharDict {
    /// Builds a dictionary from an ordered list of characters (the raw file lines),
    /// prepending the blank and optionally appending a space, exactly as
    /// `CTCLabelDecode.add_special_char` does.
    ///
    /// # Errors
    ///
    /// [`Error::Dict`] if `chars` is empty.
    pub fn from_chars(chars: Vec<String>, use_space_char: bool) -> Result<Self> {
        if chars.is_empty() {
            return Err(Error::Dict("character list is empty".into()));
        }
        let mut characters = Vec::with_capacity(chars.len() + 2);
        characters.push("blank".to_string());
        characters.extend(chars);
        if use_space_char {
            characters.push(" ".to_string());
        }
        Ok(Self { characters })
    }

    /// Parses a dictionary from file contents (one character per line).
    ///
    /// Trailing `\n` / `\r\n` are stripped per line, matching the reference reader.
    /// Empty lines are preserved as empty entries (they are legitimate dictionary
    /// slots in some ppocr dicts).
    pub fn from_str(contents: &str, use_space_char: bool) -> Result<Self> {
        let chars: Vec<String> = contents
            .split('\n')
            .map(|line| line.trim_end_matches('\r').to_string())
            .collect::<Vec<_>>();
        // The file typically ends with a trailing newline, producing one empty
        // trailing entry; drop a single trailing empty line to match Python's
        // `readlines()` + `strip` behaviour on a newline-terminated file.
        let chars = trim_trailing_empty(chars);
        Self::from_chars(chars, use_space_char)
    }

    /// Loads a dictionary from a file path.
    pub fn from_file(path: impl AsRef<Path>, use_space_char: bool) -> Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        Self::from_str(&contents, use_space_char)
    }

    /// The default PP-OCRv6 character dictionary, embedded in the binary.
    ///
    /// The v6 charset ships with the application rather than the model weight
    /// release, so it is bundled as a crate asset (`assets/ppocrv6_dict.txt`) and
    /// embedded with `include_str!`. Callers that need a different language's dict
    /// still use [`CharDict::from_file`]; this is the batteries-included default
    /// the pipeline falls back to so recognition works with no external dict file.
    ///
    /// `use_space_char` appends a space class, matching the recognizer config.
    pub fn ppocrv6(use_space_char: bool) -> Result<Self> {
        Self::from_str(
            include_str!("../assets/ppocrv6_dict.txt"),
            use_space_char,
        )
    }

    /// The number of CTC classes (including blank).
    pub fn num_classes(&self) -> usize {
        self.characters.len()
    }

    /// Maps class indices to a string, skipping any out-of-range index.
    ///
    /// The input is expected to already be collapsed and blank-free (as produced by
    /// [`mineru_burn_common::ctc::ctc_greedy_decode`]); blank (index 0) is skipped
    /// defensively in case it slips through.
    pub fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter(|&&i| i != 0)
            .filter_map(|&i| self.characters.get(i))
            .map(String::as_str)
            .collect()
    }
}

/// Drops a single trailing empty entry (from a newline-terminated file).
fn trim_trailing_empty(mut lines: Vec<String>) -> Vec<String> {
    if lines.last().map(|s| s.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_is_prepended_and_indices_offset() {
        let dict = CharDict::from_chars(vec!["a".into(), "b".into(), "c".into()], false).unwrap();
        // blank + a b c = 4 classes.
        assert_eq!(dict.num_classes(), 4);
        // class 1 -> 'a', 2 -> 'b', 3 -> 'c'.
        assert_eq!(dict.decode(&[1, 2, 3]), "abc");
    }

    #[test]
    fn space_char_appended_when_requested() {
        let dict = CharDict::from_chars(vec!["x".into()], true).unwrap();
        // blank + x + space = 3 classes; last class is the space.
        assert_eq!(dict.num_classes(), 3);
        assert_eq!(dict.decode(&[1, 2]), "x ");
    }

    #[test]
    fn blank_and_out_of_range_are_skipped() {
        let dict = CharDict::from_chars(vec!["a".into(), "b".into()], false).unwrap();
        // 0 = blank (skipped), 99 = out of range (skipped).
        assert_eq!(dict.decode(&[0, 1, 99, 2]), "ab");
    }

    #[test]
    fn parses_newline_terminated_file() {
        let dict = CharDict::from_str("a\nb\nc\n", false).unwrap();
        assert_eq!(dict.num_classes(), 4); // blank + a b c, trailing empty dropped.
        assert_eq!(dict.decode(&[1, 2, 3]), "abc");
    }

    #[test]
    fn empty_dict_is_rejected() {
        assert!(matches!(CharDict::from_chars(vec![], false), Err(Error::Dict(_))));
    }
}
