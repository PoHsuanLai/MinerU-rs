//! The output-mode selector.

/// Selects which serialized form a renderer produces.
///
/// The two Markdown modes differ only in whether visual blocks survive:
/// `mm` ("multimodal") keeps images and charts, `nlp` ("natural language")
/// drops them for a text-only corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MakeMode {
    /// Markdown with images and charts embedded.
    MmMarkdown,
    /// Text-only Markdown; images and charts are dropped.
    NlpMarkdown,
    /// The Python-compatible `content_list.json` structure.
    ContentList,
    /// The v2 `content_list.json` structure.
    ContentListV2,
}

impl MakeMode {
    /// Whether this mode keeps image/chart blocks in the output.
    ///
    /// Only [`MakeMode::NlpMarkdown`] drops them. The run flow uses this to decide
    /// whether to inject an image sink (crop-writing) at all.
    pub fn keeps_images(self) -> bool {
        !matches!(self, MakeMode::NlpMarkdown)
    }
}
