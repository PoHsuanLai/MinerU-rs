//! SLANet-plus structure-token vocabulary.
//!
//! In Python this list is read from the ONNX model's `custom_metadata_map`
//! (`character`) at load time. Because this crate hand-ports the model and never
//! opens the ONNX file, the vocabulary is embedded here as the canonical
//! PP-Structure SLANet-plus token set. It is a fixed, public list.
//!
//! [`build_vocab`] reproduces `TableLabelDecode.__init__`: it applies the
//! `merge_no_span_structure` fix-up (add `<td></td>`, drop bare `<td>`) and wraps
//! the list with the `sos` / `eos` sentinels, yielding the index space the
//! decoder argmaxes over.

/// The raw SLANet-plus structure tokens, in model output order, exactly as the
/// PP-Structure `table_structure_dict.txt` ships them (before the special-char
/// and merge fix-ups applied by [`build_vocab`]).
pub const RAW_TOKENS: &[&str] = &[
    "<thead>",
    "</thead>",
    "<tbody>",
    "</tbody>",
    "<tr>",
    "</tr>",
    "<td>",
    "<td",
    ">",
    "</td>",
    " colspan=\"2\"",
    " colspan=\"3\"",
    " colspan=\"4\"",
    " colspan=\"5\"",
    " colspan=\"6\"",
    " colspan=\"7\"",
    " colspan=\"8\"",
    " colspan=\"9\"",
    " colspan=\"10\"",
    " colspan=\"11\"",
    " colspan=\"12\"",
    " colspan=\"13\"",
    " colspan=\"14\"",
    " colspan=\"15\"",
    " colspan=\"16\"",
    " colspan=\"17\"",
    " colspan=\"18\"",
    " colspan=\"19\"",
    " rowspan=\"2\"",
    " rowspan=\"3\"",
    " rowspan=\"4\"",
    " rowspan=\"5\"",
    " rowspan=\"6\"",
    " rowspan=\"7\"",
    " rowspan=\"8\"",
    " rowspan=\"9\"",
    " rowspan=\"10\"",
    " rowspan=\"11\"",
    " rowspan=\"12\"",
    " rowspan=\"13\"",
    " rowspan=\"14\"",
    " rowspan=\"15\"",
    " rowspan=\"16\"",
    " rowspan=\"17\"",
    " rowspan=\"18\"",
    " rowspan=\"19\"",
];

/// The begin sentinel token index name.
pub const SOS: &str = "sos";
/// The end sentinel token index name.
pub const EOS: &str = "eos";

/// The decoded vocabulary: the ordered list of token strings indexed by the
/// model's output channel, plus the indices of the sentinels and the `<td>`-like
/// tokens that carry a regressed cell box.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Token string for each output-channel index.
    pub tokens: Vec<String>,
    /// Index of the `sos` (begin) sentinel.
    pub beg_idx: usize,
    /// Index of the `eos` (end) sentinel.
    pub end_idx: usize,
}

impl Vocab {
    /// True if `idx` is a `<td>`-like token that owns a regressed bbox.
    ///
    /// Mirrors `td_token = ["<td>", "<td", "<td></td>"]` in Python.
    pub fn is_td(&self, idx: usize) -> bool {
        matches!(self.tokens.get(idx).map(String::as_str), Some("<td>") | Some("<td") | Some("<td></td>"))
    }

    /// True if `idx` is one of the ignored sentinel tokens.
    pub fn is_ignored(&self, idx: usize) -> bool {
        idx == self.beg_idx || idx == self.end_idx
    }
}

/// Builds the decode vocabulary from [`RAW_TOKENS`], reproducing the Python
/// `TableLabelDecode` constructor with `merge_no_span_structure=True`.
pub fn build_vocab() -> Vocab {
    let mut chars: Vec<String> = RAW_TOKENS.iter().map(|s| s.to_string()).collect();

    // merge_no_span_structure: ensure `<td></td>` present, drop bare `<td>`.
    if !chars.iter().any(|c| c == "<td></td>") {
        chars.push("<td></td>".to_string());
    }
    chars.retain(|c| c != "<td>");

    // add_special_char: [sos] + chars + [eos].
    let mut tokens = Vec::with_capacity(chars.len() + 2);
    tokens.push(SOS.to_string());
    tokens.extend(chars);
    tokens.push(EOS.to_string());

    let beg_idx = 0;
    let end_idx = tokens.len() - 1;
    Vocab {
        tokens,
        beg_idx,
        end_idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_has_sentinels_at_ends() {
        let v = build_vocab();
        assert_eq!(v.tokens[v.beg_idx], SOS);
        assert_eq!(v.tokens[v.end_idx], EOS);
        assert_eq!(v.beg_idx, 0);
        assert_eq!(v.end_idx, v.tokens.len() - 1);
    }

    #[test]
    fn merge_no_span_structure_applied() {
        let v = build_vocab();
        assert!(v.tokens.iter().any(|t| t == "<td></td>"));
        assert!(!v.tokens.iter().any(|t| t == "<td>"));
    }

    #[test]
    fn td_and_ignored_classification() {
        let v = build_vocab();
        let td_idx = v.tokens.iter().position(|t| t == "<td></td>").unwrap();
        assert!(v.is_td(td_idx));
        assert!(v.is_ignored(v.beg_idx));
        assert!(v.is_ignored(v.end_idx));
        assert!(!v.is_ignored(td_idx));
    }
}
