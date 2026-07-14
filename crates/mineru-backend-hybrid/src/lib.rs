//! The **hybrid backend** for MinerU: interleaves the local pipeline layout model
//! with the external VLM client.
//!
//! This crate is the Rust analogue of Python's `hybrid_analyze.py` +
//! `hybrid_magic_model.py` + `hybrid_model_output_to_middle_json.py`. Its defining
//! trait — the reason "hybrid" is a distinct backend rather than VLM-with-options —
//! is that **region detection comes from the pipeline layout model, not the VLM's
//! own layout pass**. The VLM is then used only to *extract each region's content*,
//! and the pipeline OCR models *post-fill* any text the VLM left empty.
//!
//! # Design map (Python piece → Rust module)
//!
//! | Python (`mineru/backend/hybrid/…`)                              | Rust module (`crate::…`) |
//! |-----------------------------------------------------------------|--------------------------|
//! | `_validate_parse_effort`, `_resolve_effective_image_analysis`   | [`effort`] — [`Effort`] enum + validation + image-analysis branch |
//! | `MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE`, `_vlm_type_for_medium_layout_label`, `_apply_medium_visual_sub_type` | [`label_map`] — [`VlmType`] total match over [`LayoutLabel`](mineru_layout::LayoutLabel) |
//! | `MagicModel` (`hybrid_magic_model.py`), `blocks_to_page_info`, `_apply_layout_title_split`, `_normalize_split_title_blocks`, `index`→text | [`assemble`] — [`HybridAssembler`](assemble::HybridAssembler) building the typed [`Block`](mineru_types::Block) tree |
//! | `doc_analyze` / `aio_doc_analyze` orchestration, `_predict_layout_for_window`, per-region VLM extract, `_apply_post_ocr` | [`analyze`] — [`HybridBackend`] `impl Backend` |
//! | crate-wide error taxonomy                                        | [`error`] — [`Error`]/[`Result`] with `thiserror` |
//!
//! The Rust structure deliberately mirrors the sibling pipeline backend
//! (`mineru-backend-pipeline`): [`assemble`] is a pure, model-free,
//! synthetic-input-testable converter; [`analyze`] is the model-driven, PDFium-
//! serial orchestrator.
//!
//! # Composition
//!
//! [`HybridBackend::new`] takes a [`VlmClientConfig`](mineru_vlm_client::VlmClientConfig),
//! a loaded [`PipelineModels`](mineru_backend_pipeline::PipelineModels) (for the
//! layout + OCR models), and an [`Effort`]. It reuses:
//! - `mineru_layout::LayoutModel` (held inside `PipelineModels`) for region detection,
//! - `mineru_vlm_client::VlmClient` for per-region content extraction,
//! - `mineru_ocr_det` / `mineru_ocr_rec` (held inside `PipelineModels`) for post-OCR fill,
//! - `mineru_pdf` for serial rasterization.
//!
//! # Coverage: fully ported vs. simplified (be honest — this is a large port)
//!
//! **Fully ported**
//! - The `medium`/`high` effort validation and the image-analysis branch
//!   (`effort=medium` forces image analysis off).
//! - The pipeline-layout-label → VLM-extraction-type mapping (all 25 labels,
//!   including the two the Python dict omits).
//! - Region routing, per-region VLM extraction, and OCR post-fill of empty text.
//! - Block assembly: text/title/index/code/ref-text roles, discarded
//!   header/footer/page-number/aside/page-footnote, caption & footnote nesting onto
//!   the nearest visual body, `index`→text normalization, title-split by pipeline
//!   `doc_title` overlap, and inline `\(...\)` → inline-equation span splitting.
//!
//! **Simplified / deferred** (each also noted at its call site)
//! - **Per-region VLM call**: reuses [`VlmClient::extract_page`](mineru_vlm_client::VlmClient::extract_page)
//!   on each region crop rather than the reference client's
//!   `batch_extract_with_layout` (which accepts external layout blocks in one
//!   round-trip). The public client exposes no external-layout entry point, and
//!   this crate may not modify `mineru-vlm-client`. Behavior is equivalent for
//!   single-region crops; it is *not* batched and re-runs a tiny layout per crop.
//! - **Effort `high`**: shares the same per-region extraction path as `medium`
//!   rather than the Python `batch_two_step_extract`. The observable difference the
//!   port preserves is the image-analysis branch; the internal two-step vs.
//!   layout-driven distinction is collapsed because both go through the same public
//!   `extract_page`.
//! - **Formula pipeline extras**: inline-formula masking for OCR-det, MFR formula
//!   recognition sidecars, table-orientation classification, and cross-page table
//!   merging (`_process_ocr_and_formulas`, `_apply_medium_table_orientation_labels`,
//!   `cross_page_table_merge`) are **not** ported. Display equations still route to
//!   the VLM; standalone inline-formula regions are skipped as in the label map.
//! - **Paragraph merging & title-leveling finalize** (`finalize_middle_json_from_preproc`,
//!   `merge_para_text_blocks`, `apply_title_leveling_to_pdf_info`) are not run here;
//!   blocks are emitted per-region in reading order. Doc/paragraph title levels are
//!   still assigned from the layout `doc_title` overlap.
//! - **Processing-window batching** and the `_ocr_enable` (scanned-PDF) auto-classify
//!   branch are not modeled; pages are processed one at a time and text always comes
//!   from the VLM with OCR only as an empty-content fill.
//!
//! These simplifications change *completeness of enrichment*, not the shape of the
//! output: every path returns a valid [`Document`](mineru_types::Document), and no
//! path silently fabricates content — an unextractable region is left empty rather
//! than guessed.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod analyze;
pub mod assemble;
pub mod effort;
pub mod error;
pub mod label_map;

pub use analyze::HybridBackend;
pub use assemble::{AssembledPage, DocTitleBoxes, ExtractedRegion, HybridAssembler};
pub use effort::Effort;
pub use error::{Error, Result};
pub use label_map::VlmType;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use mineru_backend_pipeline::PipelineModels;
    use mineru_types::{Backend, DocInput, ParseOptions};
    use mineru_vlm_client::VlmClientConfig;

    /// A real hybrid analyze against a live VLM server + a local models directory +
    /// a demo PDF. Ignored by default: it needs an external VLM server, the pipeline
    /// model weights on disk, and a matching libpdfium. Run with:
    ///
    /// ```text
    /// MINERU_VLM_URL=http://localhost:30000/v1 \
    /// MINERU_MODELS_DIR=/path/to/PDF-Extract-Kit-1.0/models \
    /// MINERU_PDFIUM_LIB_PATH=/path/to/libpdfium.dylib \
    ///   cargo test -p mineru-backend-hybrid -- --ignored
    /// ```
    #[tokio::test]
    #[ignore = "requires a live VLM server, pipeline model weights, and libpdfium"]
    async fn analyzes_demo_pdf() {
        let base_url =
            std::env::var("MINERU_VLM_URL").unwrap_or_else(|_| "http://localhost:30000/v1".into());
        let models_dir = std::env::var("MINERU_MODELS_DIR")
            .expect("set MINERU_MODELS_DIR to the pipeline models directory");

        let config = VlmClientConfig {
            base_url,
            ..VlmClientConfig::default()
        };
        let models = PipelineModels::load(&models_dir);
        let backend = HybridBackend::new(config, models, Effort::Medium);

        let demo_dir =
            std::env::var("MINERU_DEMO_DIR").expect("set MINERU_DEMO_DIR to the demo/pdfs directory");
        let bytes = std::fs::read(std::path::Path::new(&demo_dir).join("demo1.pdf"))
            .expect("demo pdf present");
        let doc = backend
            .analyze(DocInput::new(bytes), &ParseOptions::default())
            .await
            .expect("hybrid analyze demo pdf");
        assert!(!doc.pages.is_empty());
    }
}
