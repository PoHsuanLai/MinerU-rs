//! Orchestration: PDF bytes → [`Document`], interleaving the pipeline layout model
//! with the VLM client (the `hybrid_analyze.py` analogue).
//!
//! [`HybridBackend`] opens the PDF, iterates pages **serially** (PDFium is not
//! concurrency-safe — see [`mineru_pdf`]), and for each page:
//!
//! 1. **Layout** — runs the pipeline [`LayoutModel`](mineru_layout::LayoutModel) to
//!    detect regions (the `_predict_layout_for_window` step). This *replaces* the
//!    VLM's own layout pass, which is the defining trait of the hybrid backend.
//! 2. **Route** — maps each region's label to a [`VlmType`] via the
//!    [`label_map`](crate::label_map) table (`_build_medium_vlm_layout_blocks`).
//! 3. **Extract** — for each extractable region, crops the page and asks the VLM
//!    client for that region's content, honoring the [`Effort`] image-analysis
//!    branch (`batch_extract_with_layout`).
//! 4. **Post-OCR fill** — for text regions whose VLM content came back empty, the
//!    pipeline OCR models recognize the crop and fill it in (`_apply_post_ocr`).
//! 5. **Assemble** — [`HybridAssembler`](crate::assemble::HybridAssembler) builds
//!    the typed [`Block`] tree, splitting titles via the pipeline `doc_title` boxes.
//!
//! # Coverage vs. the Python reference
//!
//! This is a faithful port of the hybrid's *essence* (pipeline-layout-driven,
//! per-region VLM extraction, OCR post-fill, effort branching, block assembly).
//! Several Python sub-paths are **deliberately simplified**, each documented at its
//! call site and summarized in the crate docs — see [`crate`]. In particular the
//! per-region VLM extraction reuses [`VlmClient::extract_page`] on each region crop
//! (the only public extraction entry point), rather than the reference client's
//! `batch_extract_with_layout` which accepts external layout blocks directly.

use async_trait::async_trait;
use image::{imageops::crop_imm, RgbImage};

use mineru_backend_pipeline::PipelineModels;
use mineru_layout::{LayoutDet, LayoutLabel};
use mineru_pdf::{PdfiumLibrary, RenderOptions};
use mineru_types::{
    Backend, BackendError, BBox, DocInput, Document, Page, PageSize, ParseOptions,
};
use mineru_vlm_client::{VlmClient, VlmClientConfig};

use crate::assemble::{DocTitleBoxes, ExtractedRegion, HybridAssembler};
use crate::effort::Effort;
use crate::error::Result;
use crate::label_map::VlmType;

/// Minimum OCR confidence for a post-OCR fill to be accepted.
///
/// Mirrors `OcrConfidence.min_confidence` in the Python `_apply_post_ocr`.
const OCR_MIN_CONFIDENCE: f32 = 0.6;

/// The hybrid parsing backend.
///
/// Owns a configured [`VlmClient`], the loaded pipeline [`PipelineModels`] (for the
/// layout model that drives region detection and the OCR models that post-fill
/// text), and the [`Effort`] knob. Cheap to share behind an `Arc`; all fields are
/// immutable after construction.
pub struct HybridBackend {
    client: VlmClient,
    models: PipelineModels,
    effort: Effort,
    image_analysis: bool,
    dpi: f32,
}

impl HybridBackend {
    /// Builds a hybrid backend from a VLM client config, loaded pipeline models, and
    /// an effort.
    ///
    /// The `models` must carry at least a layout model for region detection to
    /// occur; missing OCR models simply disable the post-OCR fill (best-effort
    /// degradation, matching the pipeline backend). `image_analysis` is the caller's
    /// requested image-analysis flag; the effort may force it off
    /// ([`Effort::effective_image_analysis`]). Rasterizes at the MinerU default
    /// 200 DPI.
    pub fn new(config: VlmClientConfig, models: PipelineModels, effort: Effort) -> Self {
        Self {
            client: VlmClient::new(config),
            models,
            effort,
            image_analysis: true,
            dpi: mineru_pdf::DEFAULT_DPI,
        }
    }

    /// Overrides the caller-requested image-analysis flag (default `true`).
    ///
    /// Under [`Effort::Medium`] this is forced off regardless; under
    /// [`Effort::High`] it is honored.
    pub fn with_image_analysis(mut self, image_analysis: bool) -> Self {
        self.image_analysis = image_analysis;
        self
    }

    /// Overrides the rasterization DPI (default 200).
    pub fn with_dpi(mut self, dpi: f32) -> Self {
        self.dpi = dpi;
        self
    }

    /// The configured effort.
    pub fn effort(&self) -> Effort {
        self.effort
    }

    /// Pixels-per-point at the configured DPI (1 point = 1/72 inch).
    fn scale(&self) -> f32 {
        self.dpi / 72.0
    }

    /// Parses the document. Rendering, layout, and VLM extraction are interleaved
    /// **serially** per page, honoring PDFium's single-threaded constraint.
    async fn run(&self, input: &DocInput, opts: &ParseOptions) -> Result<Document> {
        let lib = PdfiumLibrary::load()?;
        let doc = lib.open(&input.bytes)?;
        let render = RenderOptions::with_dpi(self.dpi);

        let (start, end) = page_bounds(doc.page_count(), opts);
        let effective_image_analysis = self.effort.effective_image_analysis(self.image_analysis);

        let mut pages = Vec::with_capacity(end.saturating_sub(start));
        for index in start..end {
            let point_size = doc.page_size(index)?;
            let image = doc.render_page(index, &render)?.into_inner();
            let page = self
                .analyze_page(index, point_size, &image, effective_image_analysis)
                .await?;
            pages.push(page);
        }

        Ok(Document { pages })
    }

    /// Runs layout → per-region VLM extraction → post-OCR fill → assembly for one
    /// already-rasterized page.
    async fn analyze_page(
        &self,
        index: usize,
        point_size: PageSize,
        image: &RgbImage,
        image_analysis: bool,
    ) -> Result<Page> {
        let scale = self.scale();

        // 1. Layout drives everything; with no layout model the page is empty.
        let dets = match &self.models.layout {
            Some(layout) => layout.detect(image)?,
            None => {
                tracing::warn!(page = index, "no layout model; hybrid emits an empty page");
                Vec::new()
            }
        };

        // Collect pipeline doc_title boxes (page pixels) for the title-split pass.
        let doc_titles = DocTitleBoxes(
            dets.iter()
                .filter(|d| d.label == LayoutLabel::DocTitle)
                .map(|d| d.bbox)
                .collect(),
        );

        // 2 + 3. Route each region to a VLM type and extract its content.
        let mut regions = Vec::with_capacity(dets.len());
        for det in dets {
            if let Some(region) = self.extract_region(det, image, image_analysis).await {
                regions.push(region);
            }
        }

        // 4. Post-OCR fill for empty text regions.
        self.post_ocr_fill(&mut regions, image);

        // Rescale region boxes from page pixels to page points for the Document.
        for region in &mut regions {
            region.bbox = scale_bbox(region.bbox, 1.0 / scale);
        }

        // 5. Assemble.
        let doc_titles = DocTitleBoxes(
            doc_titles
                .0
                .into_iter()
                .map(|b| scale_bbox(b, 1.0 / scale))
                .collect(),
        );
        let assembled = HybridAssembler::new(doc_titles).assemble(regions);

        Ok(Page {
            index,
            size: point_size,
            blocks: assembled.blocks,
            discarded: assembled.discarded,
        })
    }

    /// Routes one layout detection to a [`VlmType`] and extracts its content.
    ///
    /// Returns `None` for regions the Python skips (`inline_formula`, the reference
    /// frame). Extractable regions are cropped and sent to the VLM; visual regions
    /// (`image`/`chart`) are only content-extracted when `image_analysis` is on
    /// (matching the client's `skip_extraction`).
    async fn extract_region(
        &self,
        det: LayoutDet,
        image: &RgbImage,
        image_analysis: bool,
    ) -> Option<ExtractedRegion> {
        let vlm_type = VlmType::for_layout_label(det.label);
        if !vlm_type.is_extracted() {
            return None;
        }
        let is_seal = VlmType::visual_sub_type(det.label).is_some();

        let content = if self.should_send_to_vlm(vlm_type, image_analysis) {
            self.vlm_extract(det.bbox, image, image_analysis).await
        } else {
            None
        };

        Some(ExtractedRegion {
            bbox: det.bbox,
            vlm_type,
            content,
            order: det.order,
            is_seal,
        })
    }

    /// Whether a region's content is requested from the VLM.
    ///
    /// Image/chart bodies are only extracted when `image_analysis` is on; every
    /// other extractable type is always sent. Mirrors the client's
    /// `skip_extraction`.
    fn should_send_to_vlm(&self, vlm_type: VlmType, image_analysis: bool) -> bool {
        match vlm_type {
            VlmType::Image | VlmType::Chart => image_analysis,
            _ => true,
        }
    }

    /// Crops the region and asks the VLM client for its content.
    ///
    /// # Simplification
    ///
    /// The reference client's `batch_extract_with_layout` takes the pipeline layout
    /// as *external* blocks and extracts each in one server round-trip. The public
    /// [`VlmClient`] exposes only [`VlmClient::extract_page`], which runs its own
    /// (tiny) layout pass on the given image and then extracts. We call it on the
    /// **region crop**, so the region is re-detected within its own crop and its
    /// content extracted; we then reduce the crop's blocks to a single content
    /// string appropriate for the region's type. This preserves the hybrid's
    /// "pipeline decides *where*, VLM decides *what*" contract while staying within
    /// `mineru-vlm-client`'s public surface. A VLM error yields `None`, leaving the
    /// region to the post-OCR fill.
    async fn vlm_extract(&self, bbox: BBox, image: &RgbImage, image_analysis: bool) -> Option<String> {
        let crop = crop_region(image, &bbox)?.0;
        match self.client.extract_page(&crop, image_analysis).await {
            Ok(page) => reduce_crop_content(page),
            Err(e) => {
                tracing::warn!(error = %e, "hybrid VLM region extraction failed");
                None
            }
        }
    }

    /// Fills empty text-region content with pipeline OCR (the `_apply_post_ocr`
    /// analogue).
    ///
    /// For each text-like region whose VLM content is empty, runs OCR-det +
    /// OCR-rec over the region crop and joins the recognized lines. Only fills when
    /// the models are loaded and the recognized text clears
    /// [`OCR_MIN_CONFIDENCE`]; otherwise the region keeps its (empty) VLM content,
    /// matching the Python fallback that leaves low-confidence spans blank.
    fn post_ocr_fill(&self, regions: &mut [ExtractedRegion], image: &RgbImage) {
        let (Some(det), Some(rec)) = (&self.models.ocr_det, &self.models.ocr_rec) else {
            return; // best-effort: no OCR models -> no fill
        };

        for region in regions.iter_mut() {
            if !needs_post_ocr(region) {
                continue;
            }
            let Some((crop, _, _)) = crop_region(image, &region.bbox) else {
                continue;
            };
            let Ok(line_boxes) = det.detect(&crop) else {
                continue;
            };
            let mut parts = Vec::new();
            for line in line_boxes {
                let Some((line_crop, _, _)) = crop_region(&crop, &line) else {
                    continue;
                };
                if let Ok((text, score)) = rec.recognize(&line_crop) {
                    if score > OCR_MIN_CONFIDENCE && !text.trim().is_empty() {
                        parts.push(text);
                    }
                }
            }
            if !parts.is_empty() {
                region.content = Some(parts.join(" "));
            }
        }
    }
}

#[async_trait]
impl Backend for HybridBackend {
    async fn analyze(
        &self,
        input: DocInput,
        opts: &ParseOptions,
    ) -> std::result::Result<Document, BackendError> {
        self.run(&input, opts).await.map_err(Into::into)
    }
}

/// Whether a region should be considered for post-OCR fill: a text-like type whose
/// current content is empty.
fn needs_post_ocr(region: &ExtractedRegion) -> bool {
    let text_like = matches!(
        region.vlm_type,
        VlmType::Text
            | VlmType::Title
            | VlmType::Index
            | VlmType::RefText
            | VlmType::AsideText
            | VlmType::Header
            | VlmType::Footer
            | VlmType::PageNumber
            | VlmType::PageFootnote
            | VlmType::ImageCaption
            | VlmType::ImageFootnote
            | VlmType::Code
    );
    text_like
        && region
            .content
            .as_deref()
            .map(|c| c.trim().is_empty())
            .unwrap_or(true)
}

/// Reduces a crop's VLM blocks to one content string for the region.
///
/// A region crop usually yields one block; we take the first block carrying
/// content, falling back to joining all text spans. Table/equation content passes
/// through as-is. Returns `None` when the crop produced no content.
fn reduce_crop_content(page: mineru_vlm_client::VlmPage) -> Option<String> {
    let joined: String = page
        .blocks
        .into_iter()
        .filter_map(|b| b.content)
        .filter(|c| !c.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (!joined.trim().is_empty()).then_some(joined)
}

/// Resolves the `[start, end)` page range from options, clamped to the document.
/// Mirrors the sibling backends' bounds logic.
fn page_bounds(page_count: usize, opts: &ParseOptions) -> (usize, usize) {
    match opts.page_range {
        Some((start, end)) => {
            let start = start.min(page_count);
            let end = end.unwrap_or(page_count).min(page_count).max(start);
            (start, end)
        }
        None => (0, page_count),
    }
}

/// Crops the region from the page image, returning the crop and its `(x, y)` origin
/// in source pixel space. `None` when the box has no area or lies outside the image.
/// Mirrors the pipeline backend's `crop_region`.
fn crop_region(image: &RgbImage, bbox: &BBox) -> Option<(RgbImage, f32, f32)> {
    let (iw, ih) = (image.width() as f32, image.height() as f32);
    let x0 = bbox.x0.clamp(0.0, iw);
    let y0 = bbox.y0.clamp(0.0, ih);
    let x1 = bbox.x1.clamp(0.0, iw);
    let y1 = bbox.y1.clamp(0.0, ih);
    let w = (x1 - x0).floor() as u32;
    let h = (y1 - y0).floor() as u32;
    if w == 0 || h == 0 {
        return None;
    }
    let view = crop_imm(image, x0 as u32, y0 as u32, w, h);
    Some((view.to_image(), x0, y0))
}

/// Scales a box uniformly about the origin (pixels ↔ points).
fn scale_bbox(bbox: BBox, factor: f32) -> BBox {
    BBox::new(
        bbox.x0 * factor,
        bbox.y0 * factor,
        bbox.x1 * factor,
        bbox.y1 * factor,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_bounds_none_is_all_pages() {
        assert_eq!(page_bounds(5, &ParseOptions::default()), (0, 5));
    }

    #[test]
    fn page_bounds_clamps_range() {
        let opts = ParseOptions {
            page_range: Some((2, Some(10))),
            ..ParseOptions::default()
        };
        assert_eq!(page_bounds(5, &opts), (2, 5));
    }

    #[test]
    fn page_bounds_open_end() {
        let opts = ParseOptions {
            page_range: Some((1, None)),
            ..ParseOptions::default()
        };
        assert_eq!(page_bounds(4, &opts), (1, 4));
    }

    #[test]
    fn scale_bbox_roundtrips() {
        let b = BBox::new(72.0, 144.0, 216.0, 288.0);
        let px = scale_bbox(b, 200.0 / 72.0);
        let back = scale_bbox(px, 72.0 / 200.0);
        assert!((back.x0 - b.x0).abs() < 1e-3);
        assert!((back.y1 - b.y1).abs() < 1e-3);
    }

    #[test]
    fn crop_region_rejects_empty() {
        let img = RgbImage::new(100, 100);
        assert!(crop_region(&img, &BBox::new(10.0, 10.0, 10.0, 20.0)).is_none());
    }

    #[test]
    fn needs_post_ocr_only_for_empty_text() {
        let empty_text = ExtractedRegion {
            bbox: BBox::new(0.0, 0.0, 10.0, 10.0),
            vlm_type: VlmType::Text,
            content: Some("   ".to_owned()),
            order: 0,
            is_seal: false,
        };
        assert!(needs_post_ocr(&empty_text));

        let filled_text = ExtractedRegion {
            content: Some("hello".to_owned()),
            ..empty_text.clone()
        };
        assert!(!needs_post_ocr(&filled_text));

        let table = ExtractedRegion {
            vlm_type: VlmType::Table,
            content: None,
            ..empty_text.clone()
        };
        assert!(!needs_post_ocr(&table), "tables are not post-OCR filled");

        let image = ExtractedRegion {
            vlm_type: VlmType::Image,
            content: None,
            ..empty_text
        };
        assert!(!needs_post_ocr(&image));
    }

    #[test]
    fn reduce_crop_content_joins_nonempty() {
        use mineru_vlm_client::{VlmBlock, VlmPage};
        let page = VlmPage {
            width: 100.0,
            height: 100.0,
            blocks: vec![
                VlmBlock {
                    bbox: [0.0, 0.0, 1.0, 0.5],
                    label: "text".to_owned(),
                    content: Some("hello".to_owned()),
                    angle: 0,
                    sub_type: None,
                },
                VlmBlock {
                    bbox: [0.0, 0.5, 1.0, 1.0],
                    label: "text".to_owned(),
                    content: Some("  ".to_owned()),
                    angle: 0,
                    sub_type: None,
                },
            ],
        };
        assert_eq!(reduce_crop_content(page).as_deref(), Some("hello"));
    }

    #[test]
    fn reduce_crop_content_none_when_empty() {
        use mineru_vlm_client::VlmPage;
        let page = VlmPage {
            width: 100.0,
            height: 100.0,
            blocks: vec![],
        };
        assert!(reduce_crop_content(page).is_none());
    }
}
