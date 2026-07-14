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
//!
//! # Effort branching (`medium` vs `high`)
//!
//! The two efforts diverge in **how the VLM is driven**, mirroring the Python
//! `hybrid_analyze.py` `doc_analyze` effort branch:
//!
//! - [`Effort::Medium`] → [`HybridBackend::analyze_page_medium`]: the pipeline layout
//!   detects regions and *drives* extraction — each region crop is sent to the VLM
//!   (`batch_extract_with_layout`, where the pipeline layout is the external block
//!   list). The VLM never runs its own layout.
//! - [`Effort::High`] → [`HybridBackend::analyze_page_high`]: the VLM runs its **own**
//!   full-page two-step layout+extraction over the whole page
//!   (`batch_two_step_extract`); the pipeline layout is used *only* to collect
//!   `doc_title` boxes for the title-split pass, not to drive extraction. This is the
//!   defining behavioral difference the Python `high` effort adds.
//!
//! Both paths then feed the **same** [`HybridAssembler`], so the layout-`doc_title`
//! title split (`_apply_layout_title_split`) applies identically to either.

use async_trait::async_trait;
use image::{imageops::crop_imm, RgbImage};

use mineru_backend_pipeline::PipelineModels;
use mineru_layout::{LayoutDet, LayoutLabel};
use mineru_pdf::{PdfiumLibrary, RenderOptions};
use mineru_types::{
    Backend, BackendError, BBox, DocInput, Document, ImageWriter, Page, PageSize, ParseOptions,
};
use mineru_vlm_client::{CropSink, VlmClient, VlmClientConfig, VlmPage};

use crate::assemble::{DocTitleBoxes, ExtractedRegion, HybridAssembler};
use crate::effort::Effort;
use crate::error::Result;
use crate::label_map::VlmType;

/// Minimum OCR confidence for a post-OCR fill to be accepted.
///
/// Mirrors `OcrConfidence.min_confidence` in the Python `_apply_post_ocr`.
const OCR_MIN_CONFIDENCE: f32 = 0.6;

/// The VLM extraction seam the hybrid backend drives.
///
/// A one-method abstraction over the full-page two-step VLM call. The production
/// implementation is [`VlmClient`]; tests substitute a fake so the two effort paths
/// can be exercised without a live server (the same seam the sibling backends lack —
/// introduced here because the `high` path's routing is what needs verifying).
///
/// Both efforts call [`extract_page`](VlmExtractor::extract_page); they differ only
/// in *what image* they pass it — a single region crop (`medium`) versus the whole
/// page (`high`) — which is exactly the Python `batch_extract_with_layout` vs
/// `batch_two_step_extract` distinction expressed through the public client's only
/// extraction entry point.
#[async_trait]
trait VlmExtractor: Send + Sync {
    /// Runs the VLM's two-step layout+extraction over `image`, honoring
    /// `image_analysis` for image/chart bodies.
    ///
    /// When `sink` is `Some`, the extractor writes each visual block's crop through
    /// it (naming with `page_index`) and stamps the resulting filename onto the
    /// returned blocks' `image_ref` — this is how the `high` path (which delegates
    /// the whole page to the client) captures crops. The `medium` path passes
    /// `None` here and does its own cropping in the hybrid layer.
    async fn extract_page(
        &self,
        image: &RgbImage,
        image_analysis: bool,
        sink: Option<&dyn ImageWriter>,
        page_index: usize,
    ) -> Result<VlmPage>;
}

#[async_trait]
impl VlmExtractor for VlmClient {
    async fn extract_page(
        &self,
        image: &RgbImage,
        image_analysis: bool,
        sink: Option<&dyn ImageWriter>,
        page_index: usize,
    ) -> Result<VlmPage> {
        let crops = sink.map(|sink| CropSink { sink, page_index });
        Ok(VlmClient::extract_page(self, image, image_analysis, crops).await?)
    }
}

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
        let sink = opts.image_sink.as_deref();

        let mut pages = Vec::with_capacity(end.saturating_sub(start));
        for index in start..end {
            let point_size = doc.page_size(index)?;
            let image = doc.render_page(index, &render)?.into_inner();
            let page = self
                .analyze_page(
                    &self.client,
                    index,
                    point_size,
                    &image,
                    effective_image_analysis,
                    sink,
                )
                .await?;
            pages.push(page);
        }

        Ok(Document { pages })
    }

    /// Runs one already-rasterized page through the effort-appropriate path.
    ///
    /// Both efforts detect the pipeline layout first — `medium` uses it to *drive*
    /// extraction, `high` uses only its `doc_title` boxes for the title split — then
    /// feed the shared [`HybridAssembler`]. This is the Rust analogue of the
    /// `doc_analyze` effort branch (`hybrid_analyze.py:965-1035`).
    async fn analyze_page(
        &self,
        client: &impl VlmExtractor,
        index: usize,
        point_size: PageSize,
        image: &RgbImage,
        image_analysis: bool,
        sink: Option<&dyn ImageWriter>,
    ) -> Result<Page> {
        let scale = self.scale();

        // 1. Layout: `medium` extracts per detected region; `high` uses it only for
        //    the doc-title boxes that drive the title split.
        let dets = match &self.models.layout {
            Some(layout) => layout.detect(image)?,
            None => {
                tracing::warn!(page = index, "no layout model; hybrid emits an empty page");
                Vec::new()
            }
        };

        // Collect pipeline doc_title boxes (page pixels) for the title-split pass —
        // used by *both* efforts (`_collect_layout_doc_title_bboxes`).
        let doc_titles = DocTitleBoxes(
            dets.iter()
                .filter(|d| d.label == LayoutLabel::DocTitle)
                .map(|d| d.bbox)
                .collect(),
        );

        // 2 + 3. Extract regions, branching on effort. `medium` re-uses the pipeline
        //    layout as the block list; `high` lets the VLM run its own full-page
        //    layout and maps the resulting blocks back to regions.
        let mut regions = if self.effort.vlm_runs_own_layout() {
            self.extract_regions_high(client, index, image, image_analysis, sink)
                .await
        } else {
            self.extract_regions_medium(client, index, dets, image, image_analysis, sink)
                .await
        };

        // 4. Post-OCR fill for empty text regions (`_apply_post_ocr`). Applies to
        //    either effort's regions.
        self.post_ocr_fill(&mut regions, image);

        // Rescale region boxes from page pixels to page points for the Document.
        for region in &mut regions {
            region.bbox = scale_bbox(region.bbox, 1.0 / scale);
        }

        // 5. Assemble, applying the layout-doc_title title split to both efforts.
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

    /// **Medium effort**: the pipeline layout drives per-region VLM extraction.
    ///
    /// Each detected region is routed to a [`VlmType`] and its content extracted from
    /// its own crop — the Python `batch_extract_with_layout` shape, where the
    /// pipeline layout is the external block list and the VLM never runs its own
    /// layout pass.
    async fn extract_regions_medium(
        &self,
        client: &impl VlmExtractor,
        page_index: usize,
        dets: Vec<LayoutDet>,
        image: &RgbImage,
        image_analysis: bool,
        sink: Option<&dyn ImageWriter>,
    ) -> Vec<ExtractedRegion> {
        let mut regions = Vec::with_capacity(dets.len());
        for det in dets {
            if let Some(region) = self
                .extract_region(client, page_index, det, image, image_analysis, sink)
                .await
            {
                regions.push(region);
            }
        }
        regions
    }

    /// **High effort**: the VLM runs its own full-page two-step layout+extraction.
    ///
    /// This is the Python `batch_two_step_extract` path (`hybrid_analyze.py:1005-1033`):
    /// the *whole page* image is handed to the VLM, which detects its own layout and
    /// extracts each block, and the pipeline layout is used only for the downstream
    /// title split. The resulting VLM blocks are mapped back onto [`ExtractedRegion`]s
    /// (via [`VlmType::from_prompt_label`]) so they flow through the same assembler as
    /// the medium regions. A VLM failure yields no regions for the page (the assembler
    /// then emits an empty page), matching the Python leaving `window_model_list`
    /// empty on error.
    async fn extract_regions_high(
        &self,
        client: &impl VlmExtractor,
        page_index: usize,
        image: &RgbImage,
        image_analysis: bool,
        sink: Option<&dyn ImageWriter>,
    ) -> Vec<ExtractedRegion> {
        match client
            .extract_page(image, image_analysis, sink, page_index)
            .await
        {
            Ok(page) => vlm_page_to_regions(page),
            Err(e) => {
                tracing::warn!(error = %e, "hybrid high-effort full-page VLM extraction failed");
                Vec::new()
            }
        }
    }

    /// Routes one layout detection to a [`VlmType`] and extracts its content.
    ///
    /// Returns `None` for regions the Python skips (`inline_formula`, the reference
    /// frame). Extractable regions are cropped and sent to the VLM; visual regions
    /// (`image`/`chart`) are only content-extracted when `image_analysis` is on
    /// (matching the client's `skip_extraction`).
    async fn extract_region(
        &self,
        client: &impl VlmExtractor,
        page_index: usize,
        det: LayoutDet,
        image: &RgbImage,
        image_analysis: bool,
        sink: Option<&dyn ImageWriter>,
    ) -> Option<ExtractedRegion> {
        let vlm_type = VlmType::for_layout_label(det.label);
        if !vlm_type.is_extracted() {
            return None;
        }
        let is_seal = VlmType::visual_sub_type(det.label).is_some();

        // Visual regions (image/chart/table) get their crop written through the sink
        // in the hybrid layer — the pipeline already knows the pixel bbox. The
        // resulting filename is stamped onto the region so the assembler forwards it
        // into the body's `ImageRef` (fixing the empty `![](images)` bug).
        let image_ref = match sink {
            Some(sink) if is_visual(vlm_type) => {
                self.write_region_crop(sink, page_index, det.order, det.bbox, image)
            }
            _ => None,
        };

        let content = if self.should_send_to_vlm(vlm_type, image_analysis) {
            self.vlm_extract(client, det.bbox, image, image_analysis).await
        } else {
            None
        };

        Some(ExtractedRegion {
            bbox: det.bbox,
            vlm_type,
            content,
            order: det.order,
            is_seal,
            image_ref,
        })
    }

    /// Crops a visual region from the page raster and writes it as a PNG through the
    /// sink, returning the written filename. Best-effort: a crop that has no area or
    /// a write that fails is logged and yields `None`, leaving the ref empty.
    fn write_region_crop(
        &self,
        sink: &dyn ImageWriter,
        page_index: usize,
        order: usize,
        bbox: BBox,
        image: &RgbImage,
    ) -> Option<String> {
        let (crop, _, _) = crop_region(image, &bbox)?;
        let name = format!("p{page_index}_o{order}.png");
        match mineru_io::write_png(sink, &name, &crop) {
            Ok(()) => Some(name),
            Err(e) => {
                tracing::warn!(error = %e, name, "hybrid region crop write failed");
                None
            }
        }
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
    async fn vlm_extract(
        &self,
        client: &impl VlmExtractor,
        bbox: BBox,
        image: &RgbImage,
        image_analysis: bool,
    ) -> Option<String> {
        let crop = crop_region(image, &bbox)?.0;
        // No sink here: the medium path writes the region crop itself (in
        // `extract_region`), so the per-region content call must not re-crop.
        match client.extract_page(&crop, image_analysis, None, 0).await {
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

/// Whether a region routes to a visual body (image/chart/table) — the types whose
/// crop is written to the sink and referenced from the assembled block.
fn is_visual(vlm_type: VlmType) -> bool {
    matches!(vlm_type, VlmType::Image | VlmType::Chart | VlmType::Table)
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

/// Maps a full-page VLM result (the `high`-effort `batch_two_step_extract` output)
/// into the assembler's [`ExtractedRegion`]s.
///
/// The VLM emits blocks with **normalized** `0..1` boxes and label strings; this
/// denormalizes each box to page pixels (the space the assembler and doc-title boxes
/// share), routes the label back to a [`VlmType`] via [`VlmType::from_prompt_label`],
/// and assigns reading order from the emitted block order (the VLM returns blocks in
/// reading order, matching the pure-VLM assembler's assumption). Blocks whose label
/// routes to [`VlmType::Skipped`] are dropped, mirroring the pure-VLM assembler's
/// "unrecognized label is ignored" fallback. The `seal` sub_type the VLM may report
/// is preserved on the region so it survives the assembler's discard decision.
fn vlm_page_to_regions(page: VlmPage) -> Vec<ExtractedRegion> {
    let (w, h) = (page.width, page.height);
    page.blocks
        .into_iter()
        .enumerate()
        .filter_map(|(order, block)| {
            let vlm_type = VlmType::from_prompt_label(&block.label);
            if !vlm_type.is_extracted() {
                return None;
            }
            let [x0, y0, x1, y1] = block.bbox;
            Some(ExtractedRegion {
                bbox: BBox::new(x0 * w, y0 * h, x1 * w, y1 * h),
                vlm_type,
                content: block.content,
                order,
                is_seal: block.sub_type.as_deref() == Some("seal"),
                image_ref: block.image_ref,
            })
        })
        .collect()
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
    use mineru_backend_pipeline::PipelineModels;
    use mineru_vlm_client::{VlmBlock, VlmPage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A models bundle with no loaded models — enough to drive the effort-routing
    /// helpers, which touch the VLM seam but not `self.models` on the tested paths.
    fn empty_models() -> PipelineModels {
        PipelineModels {
            layout: None,
            ocr_det: None,
            ocr_rec: None,
            formula: None,
            table_wireless: None,
            table_wired: None,
        }
    }

    /// A hybrid backend wired to no models, at the given effort.
    fn backend(effort: Effort) -> HybridBackend {
        HybridBackend::new(VlmClientConfig::default(), empty_models(), effort)
    }

    /// A fake VLM that records how it was called and returns a canned full-page
    /// result. Used to prove the two-step (full-page) path is taken by `high` and how
    /// the per-region path is taken by `medium`, without any network.
    struct FakeVlm {
        /// The `VlmPage` returned from every `extract_page` call.
        page: VlmPage,
        /// Number of `extract_page` calls made.
        calls: AtomicUsize,
        /// The `(width, height)` of the last image passed to `extract_page`.
        last_size: std::sync::Mutex<Option<(u32, u32)>>,
    }

    impl FakeVlm {
        fn new(page: VlmPage) -> Self {
            Self {
                page,
                calls: AtomicUsize::new(0),
                last_size: std::sync::Mutex::new(None),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
        fn last_size(&self) -> Option<(u32, u32)> {
            *self.last_size.lock().unwrap()
        }
    }

    #[async_trait]
    impl VlmExtractor for FakeVlm {
        async fn extract_page(
            &self,
            image: &RgbImage,
            _image_analysis: bool,
            _sink: Option<&dyn ImageWriter>,
            _page_index: usize,
        ) -> Result<VlmPage> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last_size.lock().unwrap() = Some((image.width(), image.height()));
            Ok(self.page.clone())
        }
    }

    fn vlm_block(label: &str, content: &str, bbox: [f32; 4]) -> VlmBlock {
        VlmBlock {
            bbox,
            label: label.to_owned(),
            content: Some(content.to_owned()),
            angle: 0,
            sub_type: None,
            image_ref: None,
        }
    }

    fn full_page(blocks: Vec<VlmBlock>) -> VlmPage {
        VlmPage { width: 200.0, height: 100.0, blocks }
    }

    #[tokio::test]
    async fn high_runs_full_page_two_step_over_whole_page() {
        // The `high` path must call the VLM exactly once, on the *whole page* image
        // (not a region crop), reproducing Python's `batch_two_step_extract`.
        let page = full_page(vec![
            vlm_block("title", "Doc Title", [0.0, 0.0, 1.0, 0.1]),
            vlm_block("text", "Body.", [0.0, 0.2, 1.0, 0.4]),
        ]);
        let fake = FakeVlm::new(page);
        let img = RgbImage::new(200, 100);

        let regions = backend(Effort::High)
            .extract_regions_high(&fake, 0, &img, true, None)
            .await;

        assert_eq!(fake.calls(), 1, "high effort makes one full-page VLM call");
        assert_eq!(
            fake.last_size(),
            Some((200, 100)),
            "high effort sends the whole page, not a crop"
        );
        assert_eq!(regions.len(), 2);
        // Boxes are denormalized to page pixels.
        assert_eq!(regions[0].vlm_type, VlmType::Title);
        assert_eq!(regions[0].bbox.x1, 200.0);
        assert_eq!(regions[0].content.as_deref(), Some("Doc Title"));
        assert_eq!(regions[1].vlm_type, VlmType::Text);
    }

    #[tokio::test]
    async fn high_drops_unrecognized_and_preserves_reading_order() {
        let page = full_page(vec![
            vlm_block("text", "second", [0.0, 0.3, 1.0, 0.5]),
            vlm_block("nonsense", "?", [0.0, 0.0, 1.0, 0.1]),
            vlm_block("table", "<table></table>", [0.0, 0.6, 1.0, 0.9]),
        ]);
        let regions = vlm_page_to_regions(page);
        // The unrecognized label is dropped; the two survivors keep their emit order.
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].vlm_type, VlmType::Text);
        assert_eq!(regions[0].order, 0);
        assert_eq!(regions[1].vlm_type, VlmType::Table);
        assert_eq!(regions[1].order, 2, "order tracks the VLM's emitted index");
    }

    #[tokio::test]
    async fn high_carries_seal_sub_type() {
        let mut block = vlm_block("image", "", [0.0, 0.0, 0.5, 0.5]);
        block.sub_type = Some("seal".to_owned());
        let regions = vlm_page_to_regions(full_page(vec![block]));
        assert_eq!(regions.len(), 1);
        assert!(regions[0].is_seal);
    }

    #[tokio::test]
    async fn high_on_vlm_error_yields_no_regions() {
        // A failing VLM leaves the page empty rather than fabricating content.
        struct FailVlm;
        #[async_trait]
        impl VlmExtractor for FailVlm {
            async fn extract_page(
                &self,
                _: &RgbImage,
                _: bool,
                _: Option<&dyn ImageWriter>,
                _: usize,
            ) -> Result<VlmPage> {
                Err(crate::error::Error::Vlm(mineru_vlm_client::Error::Parse(
                    "boom".to_owned(),
                )))
            }
        }
        let img = RgbImage::new(50, 50);
        let regions = backend(Effort::High)
            .extract_regions_high(&FailVlm, 0, &img, true, None)
            .await;
        assert!(regions.is_empty());
    }

    #[tokio::test]
    async fn medium_extracts_per_region_not_full_page() {
        // The `medium` path calls the VLM once *per detected region*, each on a crop
        // — never a single full-page two-step call. Two regions => two VLM calls,
        // each smaller than the full page.
        let crop_result = full_page(vec![vlm_block("text", "region text", [0.0, 0.0, 1.0, 1.0])]);
        let fake = FakeVlm::new(crop_result);
        let img = RgbImage::new(200, 100);

        let dets = vec![
            LayoutDet { bbox: BBox::new(0.0, 0.0, 100.0, 40.0), label: LayoutLabel::Text, order: 0, score: 1.0 },
            LayoutDet { bbox: BBox::new(0.0, 50.0, 100.0, 90.0), label: LayoutLabel::Text, order: 1, score: 1.0 },
        ];
        let regions = backend(Effort::Medium)
            .extract_regions_medium(&fake, 0, dets, &img, false, None)
            .await;

        assert_eq!(fake.calls(), 2, "medium effort calls the VLM once per region");
        // The last call was on a crop, strictly smaller than the 200x100 page.
        let (w, h) = fake.last_size().expect("a crop was sent");
        assert!(w < 200 && h < 100, "medium sends region crops, not the full page");
        assert_eq!(regions.len(), 2);
    }

    /// A recording image sink for tests: captures written names without touching disk.
    #[derive(Debug, Default)]
    struct RecordingSink {
        names: std::sync::Mutex<Vec<String>>,
    }
    impl mineru_types::ImageWriter for RecordingSink {
        fn write(&self, name: &str, _bytes: &[u8]) -> std::io::Result<()> {
            self.names
                .lock()
                .map_err(|_| std::io::Error::other("poisoned"))?
                .push(name.to_owned());
            Ok(())
        }
    }

    #[tokio::test]
    async fn medium_stamps_image_ref_from_written_crop() {
        // Guards the `![](images)` bug on the hybrid medium path: when a sink is
        // present, an image region's crop is written and its filename flows all the
        // way into a non-empty `ImageRef` on the assembled block.
        use mineru_types::{Block, ImageBody, ImageRef};

        let fake = FakeVlm::new(full_page(vec![]));
        let sink = RecordingSink::default();
        let img = RgbImage::new(200, 100);

        let dets = vec![LayoutDet {
            bbox: BBox::new(10.0, 10.0, 90.0, 90.0),
            label: LayoutLabel::Image,
            order: 3,
            score: 1.0,
        }];
        let regions = backend(Effort::Medium)
            .extract_regions_medium(&fake, 7, dets, &img, true, Some(&sink))
            .await;

        // The crop was written under the `p{page}_o{order}.png` name.
        assert_eq!(*sink.names.lock().unwrap(), vec!["p7_o3.png".to_owned()]);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].image_ref.as_deref(), Some("p7_o3.png"));

        // And it survives assembly into a non-empty ImageRef (not the empty-string bug).
        let assembled = HybridAssembler::default().assemble(regions);
        match &assembled.blocks[0] {
            Block::Image(c) => {
                let ImageBody { image: ImageRef(r), .. } = &c.body;
                assert_eq!(r, "p7_o3.png", "image_ref must reach the assembled body");
            }
            other => panic!("expected image block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn medium_without_sink_leaves_image_ref_empty() {
        // No sink → no crop written, ref stays empty (the acceptable degraded case).
        let fake = FakeVlm::new(full_page(vec![]));
        let img = RgbImage::new(200, 100);
        let dets = vec![LayoutDet {
            bbox: BBox::new(10.0, 10.0, 90.0, 90.0),
            label: LayoutLabel::Image,
            order: 0,
            score: 1.0,
        }];
        let regions = backend(Effort::Medium)
            .extract_regions_medium(&fake, 0, dets, &img, true, None)
            .await;
        assert_eq!(regions.len(), 1);
        assert!(regions[0].image_ref.is_none());
    }

    #[tokio::test]
    async fn effort_selects_distinct_paths() {
        // High makes exactly one full-page call; Medium makes none for zero regions
        // (proving Medium never issues the single full-page two-step call High does).
        let fake_high = FakeVlm::new(full_page(vec![vlm_block("text", "x", [0.0, 0.0, 1.0, 1.0])]));
        let img = RgbImage::new(120, 80);
        backend(Effort::High).extract_regions_high(&fake_high, 0, &img, true, None).await;
        assert_eq!(fake_high.calls(), 1);

        let fake_medium = FakeVlm::new(full_page(vec![]));
        let no_regions = backend(Effort::Medium)
            .extract_regions_medium(&fake_medium, 0, Vec::new(), &img, false, None)
            .await;
        assert_eq!(fake_medium.calls(), 0, "medium with no regions makes no VLM call");
        assert!(no_regions.is_empty());
    }

    #[test]
    fn vlm_page_to_regions_denormalizes_boxes() {
        let page = full_page(vec![vlm_block("text", "t", [0.1, 0.2, 0.6, 0.8])]);
        let regions = vlm_page_to_regions(page);
        assert_eq!(regions.len(), 1);
        let b = regions[0].bbox;
        assert!((b.x0 - 20.0).abs() < 1e-3 && (b.y0 - 20.0).abs() < 1e-3);
        assert!((b.x1 - 120.0).abs() < 1e-3 && (b.y1 - 80.0).abs() < 1e-3);
    }

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
            image_ref: None,
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
                    image_ref: None,
                },
                VlmBlock {
                    bbox: [0.0, 0.5, 1.0, 1.0],
                    label: "text".to_owned(),
                    content: Some("  ".to_owned()),
                    angle: 0,
                    sub_type: None,
                    image_ref: None,
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
