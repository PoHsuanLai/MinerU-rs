//! The pipeline-layout-label → VLM-extraction-type mapping.
//!
//! Python reference: `hybrid_analyze.py:83-107`,
//! `MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE`, consumed by
//! `_vlm_type_for_medium_layout_label` when building the VLM's external layout
//! blocks. It maps each PP-DocLayoutV2 layout label (a
//! [`LayoutLabel`](mineru_layout::LayoutLabel)) to the block *type* the
//! `mineru-vl-utils` VLM understands for content extraction.
//!
//! In the Python this dict is stringly-typed (`"abstract" -> BlockType.TEXT`,
//! …). Here it is a total match from the typed [`LayoutLabel`] to a
//! payload-carrying [`VlmType`] enum, so an unmapped label is a compile-time
//! omission rather than a silent `dict.get() -> None`. Labels the Python dict
//! omits (they carry no `.get` entry, so `_vlm_type_for_medium_layout_label`
//! returns `None` and the region is skipped) map to [`VlmType::Skipped`].

use mineru_layout::LayoutLabel;

/// The VLM extraction type a layout region is routed to.
///
/// Mirrors the `mineru-vl-utils` `BlockType` values the Python maps each layout
/// label onto. Each variant names the block type the VLM extracts as; the
/// [`Skipped`](VlmType::Skipped) variant marks labels the Python dict leaves out
/// (so the region is not sent to the VLM at all).
///
/// The [`prompt_label`](VlmType::prompt_label) of each variant is the string the
/// VLM client uses to pick its extraction prompt and to tag the resulting block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlmType {
    /// Flowing text (`BlockType.TEXT`).
    Text,
    /// A title (`BlockType.TITLE`) — later split into doc/paragraph title by the
    /// layout-title-split pass.
    Title,
    /// A table-of-contents / index block (`BlockType.INDEX`); normalized back to
    /// plain text on output.
    Index,
    /// A code block (`BlockType.CODE`).
    Code,
    /// A reference-list item (`BlockType.REF_TEXT`).
    RefText,
    /// Marginal aside text (`BlockType.ASIDE_TEXT`).
    AsideText,
    /// Page header (`BlockType.HEADER`).
    Header,
    /// Page footer (`BlockType.FOOTER`).
    Footer,
    /// Page number (`BlockType.PAGE_NUMBER`).
    PageNumber,
    /// Page footnote (`BlockType.PAGE_FOOTNOTE`).
    PageFootnote,
    /// A formula number / label (`BlockType.FORMULA_NUMBER`).
    FormulaNumber,
    /// An image caption (`BlockType.IMAGE_CAPTION`).
    ImageCaption,
    /// An image footnote (`BlockType.IMAGE_FOOTNOTE`).
    ImageFootnote,
    /// An image body (`BlockType.IMAGE`).
    Image,
    /// A chart body (`BlockType.CHART`).
    Chart,
    /// A table body (`BlockType.TABLE`).
    Table,
    /// A display equation (`BlockType.EQUATION`).
    Equation,
    /// A label the Python `MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE` dict does not
    /// contain: the region is not routed to the VLM (Python
    /// `_vlm_type_for_medium_layout_label` returns `None`).
    Skipped,
}

impl VlmType {
    /// Maps a pipeline layout label to its VLM extraction type.
    ///
    /// This is the Rust-native transcription of
    /// `MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE`. The match is *total* over
    /// [`LayoutLabel`]; labels the Python dict omits
    /// (`inline_formula`, `reference` frame, `content` maps to `Index`) return
    /// [`VlmType::Skipped`] where the Python `.get` would return `None`.
    pub fn for_layout_label(label: LayoutLabel) -> Self {
        use LayoutLabel as L;
        match label {
            L::Abstract => VlmType::Text,
            L::Algorithm => VlmType::Code,
            L::AsideText => VlmType::AsideText,
            L::Content => VlmType::Index,
            L::DocTitle => VlmType::Title,
            L::Footer => VlmType::Footer,
            L::FooterImage => VlmType::Footer,
            L::Footnote => VlmType::PageFootnote,
            L::FormulaNumber => VlmType::FormulaNumber,
            L::Header => VlmType::Header,
            L::HeaderImage => VlmType::Header,
            L::Number => VlmType::PageNumber,
            L::ParagraphTitle => VlmType::Title,
            L::ReferenceContent => VlmType::RefText,
            L::Text => VlmType::Text,
            L::VerticalText => VlmType::Text,
            L::FigureTitle => VlmType::ImageCaption,
            L::VisionFootnote => VlmType::ImageFootnote,
            L::Image => VlmType::Image,
            L::Chart => VlmType::Chart,
            L::Seal => VlmType::Image,
            L::Table => VlmType::Table,
            L::DisplayFormula => VlmType::Equation,

            // Not present in MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE — skipped by
            // `_vlm_type_for_medium_layout_label` returning None.
            L::InlineFormula | L::Reference => VlmType::Skipped,
        }
    }

    /// Whether this region is routed to the VLM at all.
    pub fn is_extracted(self) -> bool {
        !matches!(self, VlmType::Skipped)
    }

    /// Maps a VLM block-label string back to a [`VlmType`].
    ///
    /// The inverse of [`prompt_label`](VlmType::prompt_label), used by the
    /// `high`-effort path: the full-page VLM (`batch_two_step_extract`) returns
    /// blocks tagged with `mineru-vl-utils` `BlockType.value` strings, which we route
    /// back through the same assembler as the pipeline-driven `medium` regions. Any
    /// label outside the recognized vocabulary — including `image_block`, which the
    /// VLM emits as an alias for `image` — is folded to the nearest type;
    /// truly unknown labels become [`VlmType::Skipped`] so they drop from the tree,
    /// matching the pure-VLM assembler's "unrecognized label is ignored" fallback.
    pub fn from_prompt_label(label: &str) -> Self {
        match label {
            "text" | "phonetic" => VlmType::Text,
            "title" => VlmType::Title,
            "index" => VlmType::Index,
            "code" | "algorithm" => VlmType::Code,
            "ref_text" => VlmType::RefText,
            "aside_text" => VlmType::AsideText,
            "header" => VlmType::Header,
            "footer" => VlmType::Footer,
            "page_number" => VlmType::PageNumber,
            "page_footnote" => VlmType::PageFootnote,
            "formula_number" => VlmType::FormulaNumber,
            "image_caption" | "table_caption" | "code_caption" => VlmType::ImageCaption,
            "image_footnote" | "table_footnote" => VlmType::ImageFootnote,
            "image" | "image_block" => VlmType::Image,
            "chart" => VlmType::Chart,
            "table" => VlmType::Table,
            "equation" => VlmType::Equation,
            _ => VlmType::Skipped,
        }
    }

    /// The `sub_type` the Python `_apply_medium_visual_sub_type` pins for a `seal`
    /// layout label, else `None`.
    ///
    /// Only the `seal` label carries a `sub_type` ("seal"); every other label maps
    /// to `None`. Kept here (keyed on the *label*, not the type) because two labels
    /// (`image`, `seal`) both map to [`VlmType::Image`] but only `seal` is tagged.
    pub fn visual_sub_type(label: LayoutLabel) -> Option<&'static str> {
        match label {
            LayoutLabel::Seal => Some("seal"),
            _ => None,
        }
    }

    /// The label string the VLM client uses to select its extraction prompt and to
    /// tag the emitted block (the `mineru-vl-utils` `BlockType.value`).
    ///
    /// These strings feed `mineru_vlm_client`'s `extraction_prompt` / assembly,
    /// which recognizes `"text"`, `"title"`, `"image"`, `"chart"`, `"table"`,
    /// `"equation"`, `"code"`, and the caption/footnote/discarded roles.
    pub fn prompt_label(self) -> &'static str {
        match self {
            VlmType::Text => "text",
            VlmType::Title => "title",
            VlmType::Index => "index",
            VlmType::Code => "code",
            VlmType::RefText => "ref_text",
            VlmType::AsideText => "aside_text",
            VlmType::Header => "header",
            VlmType::Footer => "footer",
            VlmType::PageNumber => "page_number",
            VlmType::PageFootnote => "page_footnote",
            VlmType::FormulaNumber => "formula_number",
            VlmType::ImageCaption => "image_caption",
            VlmType::ImageFootnote => "image_footnote",
            VlmType::Image => "image",
            VlmType::Chart => "chart",
            VlmType::Table => "table",
            VlmType::Equation => "equation",
            // Skipped regions are never sent; give a stable inert label.
            VlmType::Skipped => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mineru_layout::LayoutLabel as L;

    #[test]
    fn maps_core_text_labels() {
        assert_eq!(VlmType::for_layout_label(L::Text), VlmType::Text);
        assert_eq!(VlmType::for_layout_label(L::Abstract), VlmType::Text);
        assert_eq!(VlmType::for_layout_label(L::VerticalText), VlmType::Text);
    }

    #[test]
    fn maps_titles_and_content() {
        assert_eq!(VlmType::for_layout_label(L::DocTitle), VlmType::Title);
        assert_eq!(VlmType::for_layout_label(L::ParagraphTitle), VlmType::Title);
        // `content` -> INDEX (Python), later normalized to text on output.
        assert_eq!(VlmType::for_layout_label(L::Content), VlmType::Index);
    }

    #[test]
    fn maps_visual_and_formula_labels() {
        assert_eq!(VlmType::for_layout_label(L::Image), VlmType::Image);
        assert_eq!(VlmType::for_layout_label(L::Seal), VlmType::Image);
        assert_eq!(VlmType::for_layout_label(L::Chart), VlmType::Chart);
        assert_eq!(VlmType::for_layout_label(L::Table), VlmType::Table);
        assert_eq!(VlmType::for_layout_label(L::DisplayFormula), VlmType::Equation);
        assert_eq!(VlmType::for_layout_label(L::Algorithm), VlmType::Code);
    }

    #[test]
    fn maps_discarded_and_caption_labels() {
        assert_eq!(VlmType::for_layout_label(L::Header), VlmType::Header);
        assert_eq!(VlmType::for_layout_label(L::HeaderImage), VlmType::Header);
        assert_eq!(VlmType::for_layout_label(L::Footer), VlmType::Footer);
        assert_eq!(VlmType::for_layout_label(L::FooterImage), VlmType::Footer);
        assert_eq!(VlmType::for_layout_label(L::Number), VlmType::PageNumber);
        assert_eq!(VlmType::for_layout_label(L::Footnote), VlmType::PageFootnote);
        assert_eq!(VlmType::for_layout_label(L::FigureTitle), VlmType::ImageCaption);
        assert_eq!(VlmType::for_layout_label(L::VisionFootnote), VlmType::ImageFootnote);
        assert_eq!(VlmType::for_layout_label(L::ReferenceContent), VlmType::RefText);
        assert_eq!(VlmType::for_layout_label(L::AsideText), VlmType::AsideText);
        assert_eq!(VlmType::for_layout_label(L::FormulaNumber), VlmType::FormulaNumber);
    }

    #[test]
    fn omitted_labels_are_skipped() {
        // Python dict has no entry for these -> None -> region skipped.
        assert_eq!(VlmType::for_layout_label(L::InlineFormula), VlmType::Skipped);
        assert_eq!(VlmType::for_layout_label(L::Reference), VlmType::Skipped);
        assert!(!VlmType::for_layout_label(L::InlineFormula).is_extracted());
        assert!(VlmType::for_layout_label(L::Text).is_extracted());
    }

    #[test]
    fn seal_carries_sub_type() {
        assert_eq!(VlmType::visual_sub_type(L::Seal), Some("seal"));
        assert_eq!(VlmType::visual_sub_type(L::Image), None);
        assert_eq!(VlmType::visual_sub_type(L::Chart), None);
    }

    #[test]
    fn from_prompt_label_inverts_prompt_label() {
        // Round-trips every non-skipped type through prompt_label -> from_prompt_label.
        for ty in [
            VlmType::Text,
            VlmType::Title,
            VlmType::Index,
            VlmType::Code,
            VlmType::RefText,
            VlmType::AsideText,
            VlmType::Header,
            VlmType::Footer,
            VlmType::PageNumber,
            VlmType::PageFootnote,
            VlmType::FormulaNumber,
            VlmType::ImageCaption,
            VlmType::ImageFootnote,
            VlmType::Image,
            VlmType::Chart,
            VlmType::Table,
            VlmType::Equation,
        ] {
            assert_eq!(VlmType::from_prompt_label(ty.prompt_label()), ty);
        }
    }

    #[test]
    fn from_prompt_label_folds_aliases_and_unknowns() {
        // Aliases the full-page VLM can emit fold to their canonical type.
        assert_eq!(VlmType::from_prompt_label("image_block"), VlmType::Image);
        assert_eq!(VlmType::from_prompt_label("algorithm"), VlmType::Code);
        assert_eq!(VlmType::from_prompt_label("table_caption"), VlmType::ImageCaption);
        // Truly unknown labels drop out of the tree.
        assert_eq!(VlmType::from_prompt_label("nonsense"), VlmType::Skipped);
        assert_eq!(VlmType::from_prompt_label(""), VlmType::Skipped);
    }

    #[test]
    fn prompt_labels_match_vlm_client_vocab() {
        // The strings the VLM client's assemble/prompt layer recognizes.
        assert_eq!(VlmType::Text.prompt_label(), "text");
        assert_eq!(VlmType::Title.prompt_label(), "title");
        assert_eq!(VlmType::Table.prompt_label(), "table");
        assert_eq!(VlmType::Equation.prompt_label(), "equation");
        assert_eq!(VlmType::Image.prompt_label(), "image");
        assert_eq!(VlmType::Code.prompt_label(), "code");
    }

    #[test]
    fn every_layout_label_maps_totally() {
        // Exhaustiveness guard: for_layout_label is a total match, so this simply
        // exercises all 25 labels without panicking and asserts skips are minimal.
        let skipped = mineru_layout::label::ALL_LABELS
            .iter()
            .filter(|&&l| VlmType::for_layout_label(l) == VlmType::Skipped)
            .count();
        assert_eq!(skipped, 2, "only inline_formula + reference frame are skipped");
    }
}
