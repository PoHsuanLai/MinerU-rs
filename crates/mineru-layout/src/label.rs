//! The PP-DocLayoutV2 layout-class taxonomy.
//!
//! Mirrors `PP_DOCLAYOUT_V2_LABELS`, `DEFAULT_CLASS_THRESHOLDS`, and
//! `DEFAULT_CLASS_ORDER` from the Python reference (`pp_doclayoutv2.py`). The
//! class id ordering is load-bearing: it is the index space of the detector's
//! class logits, so it must match the checkpoint exactly.

use crate::error::{Error, Result};

/// A PP-DocLayoutV2 layout class.
///
/// The discriminant of each variant equals its class id in the model's output
/// logits (0-based, matching `PP_DOCLAYOUT_V2_LABELS`). There are 25 classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LayoutLabel {
    /// Paper abstract (论文摘要).
    Abstract = 0,
    /// Algorithm block (算法).
    Algorithm = 1,
    /// Marginal / aside text near the page edge (页边注文本).
    AsideText = 2,
    /// Chart / data visualization (图表).
    Chart = 3,
    /// Table-of-contents content block (目录块).
    Content = 4,
    /// A standalone display formula (独立展示公式).
    DisplayFormula = 5,
    /// Document title (文章标题).
    DocTitle = 6,
    /// Caption for an image/chart/table (图题).
    FigureTitle = 7,
    /// Page footer text (页脚文本).
    Footer = 8,
    /// Page footer image (页脚图片).
    FooterImage = 9,
    /// Page footnote (脚注).
    Footnote = 10,
    /// Formula number / label (公式编号).
    FormulaNumber = 11,
    /// Page header text (页眉文本).
    Header = 12,
    /// Page header image (页眉图片).
    HeaderImage = 13,
    /// Image (图片).
    Image = 14,
    /// Inline formula (行内公式).
    InlineFormula = 15,
    /// Page number (页码).
    Number = 16,
    /// Paragraph title, distinct from the document title (段落标题).
    ParagraphTitle = 17,
    /// Reference-list outer frame (参考文献外框).
    Reference = 18,
    /// A single reference-list item (参考文献内容).
    ReferenceContent = 19,
    /// Seal / stamp (印章).
    Seal = 20,
    /// Table (表格).
    Table = 21,
    /// General body text (一般文本).
    Text = 22,
    /// Vertically-set text (竖排文本).
    VerticalText = 23,
    /// Footnote attached to an image/chart/table (视觉脚注).
    VisionFootnote = 24,
}

/// Number of layout classes.
pub const NUM_CLASSES: usize = 25;

/// All labels in class-id order. Index `i` holds the label with id `i`.
pub const ALL_LABELS: [LayoutLabel; NUM_CLASSES] = [
    LayoutLabel::Abstract,
    LayoutLabel::Algorithm,
    LayoutLabel::AsideText,
    LayoutLabel::Chart,
    LayoutLabel::Content,
    LayoutLabel::DisplayFormula,
    LayoutLabel::DocTitle,
    LayoutLabel::FigureTitle,
    LayoutLabel::Footer,
    LayoutLabel::FooterImage,
    LayoutLabel::Footnote,
    LayoutLabel::FormulaNumber,
    LayoutLabel::Header,
    LayoutLabel::HeaderImage,
    LayoutLabel::Image,
    LayoutLabel::InlineFormula,
    LayoutLabel::Number,
    LayoutLabel::ParagraphTitle,
    LayoutLabel::Reference,
    LayoutLabel::ReferenceContent,
    LayoutLabel::Seal,
    LayoutLabel::Table,
    LayoutLabel::Text,
    LayoutLabel::VerticalText,
    LayoutLabel::VisionFootnote,
];

/// Per-class confidence thresholds (`DEFAULT_CLASS_THRESHOLDS`), indexed by class
/// id. Applied before reading-order decoding in the model's forward pass.
pub const CLASS_THRESHOLDS: [f32; NUM_CLASSES] = [
    0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.5, 0.4, 0.5,
    0.5, 0.45, 0.5, 0.4, 0.4, 0.5,
];

/// Reading-order class remap (`DEFAULT_CLASS_ORDER`), indexed by detection class
/// id, producing the ~16-way category id fed to the reading-order head's label
/// embedding. Length 25; values are in `0..reading_order_config.num_classes`.
pub const CLASS_ORDER: [i64; NUM_CLASSES] = [
    4, 2, 14, 1, 5, 7, 8, 6, 11, 11, 9, 13, 10, 10, 1, 2, 3, 0, 2, 2, 12, 1, 2, 15, 6,
];

impl LayoutLabel {
    /// The class id (0-based) of this label.
    pub fn id(self) -> usize {
        self as usize
    }

    /// The label for a class id, or [`Error::Config`] if out of range.
    pub fn from_id(id: usize) -> Result<Self> {
        ALL_LABELS
            .get(id)
            .copied()
            .ok_or_else(|| Error::Config(format!("layout class id {id} out of range (max {NUM_CLASSES})")))
    }

    /// The confidence threshold for this class.
    pub fn threshold(self) -> f32 {
        CLASS_THRESHOLDS[self.id()]
    }

    /// The snake_case name used by the Python reference and the `middle_json`.
    pub fn name(self) -> &'static str {
        match self {
            LayoutLabel::Abstract => "abstract",
            LayoutLabel::Algorithm => "algorithm",
            LayoutLabel::AsideText => "aside_text",
            LayoutLabel::Chart => "chart",
            LayoutLabel::Content => "content",
            LayoutLabel::DisplayFormula => "display_formula",
            LayoutLabel::DocTitle => "doc_title",
            LayoutLabel::FigureTitle => "figure_title",
            LayoutLabel::Footer => "footer",
            LayoutLabel::FooterImage => "footer_image",
            LayoutLabel::Footnote => "footnote",
            LayoutLabel::FormulaNumber => "formula_number",
            LayoutLabel::Header => "header",
            LayoutLabel::HeaderImage => "header_image",
            LayoutLabel::Image => "image",
            LayoutLabel::InlineFormula => "inline_formula",
            LayoutLabel::Number => "number",
            LayoutLabel::ParagraphTitle => "paragraph_title",
            LayoutLabel::Reference => "reference",
            LayoutLabel::ReferenceContent => "reference_content",
            LayoutLabel::Seal => "seal",
            LayoutLabel::Table => "table",
            LayoutLabel::Text => "text",
            LayoutLabel::VerticalText => "vertical_text",
            LayoutLabel::VisionFootnote => "vision_footnote",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_id_roundtrips() {
        for (i, &label) in ALL_LABELS.iter().enumerate() {
            assert_eq!(label.id(), i);
            assert_eq!(LayoutLabel::from_id(i).expect("valid id"), label);
        }
    }

    #[test]
    fn from_id_rejects_out_of_range() {
        assert!(LayoutLabel::from_id(NUM_CLASSES).is_err());
    }

    #[test]
    fn thresholds_match_python_reference() {
        // Spot-check the non-0.5 entries from DEFAULT_CLASS_THRESHOLDS.
        assert_eq!(LayoutLabel::DisplayFormula.threshold(), 0.4);
        assert_eq!(LayoutLabel::DocTitle.threshold(), 0.4);
        assert_eq!(LayoutLabel::InlineFormula.threshold(), 0.4);
        assert_eq!(LayoutLabel::ParagraphTitle.threshold(), 0.4);
        assert_eq!(LayoutLabel::Seal.threshold(), 0.45);
        assert_eq!(LayoutLabel::Text.threshold(), 0.4);
        assert_eq!(LayoutLabel::VerticalText.threshold(), 0.4);
        assert_eq!(LayoutLabel::Table.threshold(), 0.5);
    }

    #[test]
    fn class_order_has_expected_length_and_range() {
        assert_eq!(CLASS_ORDER.len(), NUM_CLASSES);
        assert!(CLASS_ORDER.iter().all(|&c| (0..16).contains(&c)));
        // paragraph_title (17) maps to reading-order category 0.
        assert_eq!(CLASS_ORDER[LayoutLabel::ParagraphTitle.id()], 0);
    }

    #[test]
    fn names_are_snake_case_reference() {
        assert_eq!(LayoutLabel::AsideText.name(), "aside_text");
        assert_eq!(LayoutLabel::DisplayFormula.name(), "display_formula");
        assert_eq!(LayoutLabel::VisionFootnote.name(), "vision_footnote");
    }
}
