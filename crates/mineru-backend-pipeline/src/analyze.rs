//! Orchestration: PDF bytes → [`Document`] (the `pipeline_analyze.py` analogue).
//!
//! [`PipelineBackend`] opens the PDF, iterates pages **serially** (PDFium is not
//! concurrency-safe — see [`mineru_pdf`]), rasterizes each at 200 DPI, extracts the
//! native (embedded) text layer, runs layout detection, then per-region recognition
//! (native text-fill / OCR / formula / table), and hands the raw regions to the
//! [`PageAssembler`](crate::assemble::PageAssembler) which builds the typed
//! [`Block`] tree. A light [`para`](crate::para) pass merges adjacent paragraphs.
//!
//! # Native text vs OCR (the digital/scanned split)
//!
//! For each text region we first try to fill it from the page's embedded text layer
//! ([`mineru_pdf::PdfDocument::extract_text`], grouped by the ported `pdftext`
//! heuristics). Digital PDFs read their text layer directly — fast and exact — and
//! only regions the native layer cannot fill (a scanned page has *no* embedded text,
//! so every region is unfillable) fall through to the OCR det+rec path, which behaves
//! exactly as before. This is a per-region simplification of Python's document-level
//! `pdf_classify` (auto/txt/ocr): rather than classifying the whole doc, each region
//! is filled-or-OCR'd on its own. Scanned PDFs therefore still OCR every line.
//!
//! Recognition is *best-effort per stage*: a region whose model is unloaded (see
//! [`PipelineModels`](crate::PipelineModels)) still produces a block, just without
//! its recognized text/latex/html — the layout structure is always emitted.

use async_trait::async_trait;
use image::{imageops::crop_imm, RgbImage};

use mineru_layout::LayoutDet;
use mineru_pdf::{PageText, PdfiumLibrary, RenderOptions};
use mineru_types::{
    Backend, BackendError, BBox, DocInput, Document, ImageRef, Page, PageSize, ParseOptions,
};

use crate::assemble::{PageAssembler, RecognizedLine, Region, RegionContent, RegionKind};
use crate::models::PipelineModels;
use crate::para::merge_paragraphs;

/// The local Burn-model pipeline backend.
///
/// Owns the loaded [`PipelineModels`] and implements
/// [`Backend`](mineru_types::Backend). Cheap to share behind an `Arc`; the models
/// are immutable after construction.
pub struct PipelineBackend {
    models: PipelineModels,
    dpi: f32,
}

impl PipelineBackend {
    /// Builds a backend from already-loaded models, rasterizing at the MinerU
    /// default 200 DPI.
    pub fn new(models: PipelineModels) -> Self {
        Self {
            models,
            dpi: mineru_pdf::DEFAULT_DPI,
        }
    }

    /// Overrides the rasterization DPI (default 200).
    pub fn with_dpi(mut self, dpi: f32) -> Self {
        self.dpi = dpi;
        self
    }

    /// Pixels-per-point at the configured DPI (1 point = 1/72 inch).
    fn scale(&self) -> f32 {
        self.dpi / 72.0
    }

    /// Parses the document, converting any internal error to a [`BackendError`].
    fn run(&self, input: &DocInput, opts: &ParseOptions) -> crate::Result<Document> {
        let lib = PdfiumLibrary::load()?;
        let doc = lib.open(&input.bytes)?;
        let render = RenderOptions::with_dpi(self.dpi);

        let (start, end) = page_bounds(doc.page_count(), opts);
        let mut pages = Vec::with_capacity(end.saturating_sub(start));

        // SERIAL iteration: PDFium is not safe for concurrent page ops.
        for index in start..end {
            let size = doc.page_size(index)?;
            let image = doc.render_page(index, &render)?.into_inner();
            // Native text layer for this page (empty for scanned pages). A read
            // failure is non-fatal: fall back to OCR for the whole page.
            let page_text = doc.extract_text(index).unwrap_or_else(|e| {
                tracing::warn!(page = index, error = %e, "native text extraction failed");
                PageText::default()
            });
            let page = self.analyze_page(index, size, &image, &page_text);
            pages.push(page);
        }

        Ok(Document { pages })
    }

    /// Runs layout + recognition + assembly for one already-rasterized page.
    fn analyze_page(
        &self,
        index: usize,
        size: PageSize,
        image: &RgbImage,
        page_text: &PageText,
    ) -> Page {
        let scale = self.scale();

        // Layout is the driver; with no layout model the page is empty.
        let dets = match &self.models.layout {
            Some(layout) => layout.detect(image).unwrap_or_else(|e| {
                tracing::warn!(page = index, error = %e, "layout detect failed");
                Vec::new()
            }),
            None => Vec::new(),
        };

        let regions: Vec<Region> = dets
            .into_iter()
            .map(|det| self.recognize_region(index, det, image, scale, page_text))
            .collect();

        let assembled = PageAssembler.assemble(regions);
        let blocks = merge_paragraphs(assembled.blocks);

        Page {
            index,
            size,
            blocks,
            discarded: assembled.discarded,
        }
    }

    /// Runs the recognition model that a region's kind calls for, scaling the
    /// detection box from pixels to page points and returning the region for
    /// assembly.
    fn recognize_region(
        &self,
        page: usize,
        det: LayoutDet,
        image: &RgbImage,
        scale: f32,
        page_text: &PageText,
    ) -> Region {
        let pixel_bbox = det.bbox;
        // Detection box in page points (the space native text lives in).
        let point_bbox = scale_bbox(pixel_bbox, 1.0 / scale);
        let content = match RegionKind::classify(det.label) {
            RegionKind::Text(_) | RegionKind::Caption | RegionKind::Footnote => {
                self.recognize_text(&pixel_bbox, &point_bbox, image, scale, page_text)
            }
            RegionKind::Discarded(_) => {
                self.recognize_text(&pixel_bbox, &point_bbox, image, scale, page_text)
            }
            RegionKind::Equation => self.recognize_formula(&pixel_bbox, image),
            RegionKind::Table => self.recognize_table(&pixel_bbox, image),
            RegionKind::Image | RegionKind::Chart => self.crop_image(page, &det, image),
            RegionKind::Ignored => RegionContent::default(),
        };

        // Rescale the detection box to page points for the assembled Document.
        Region {
            det: LayoutDet {
                bbox: point_bbox,
                ..det
            },
            content,
        }
    }

    /// Recognizes a text region: native text-fill first, OCR as the fallback.
    ///
    /// If the page's embedded text layer can fill this region's box
    /// ([`PageText::fill_regions`]), the native text is used as a single recognized
    /// line (exact, no model needed). Otherwise — no embedded text (scanned page),
    /// or a box the native layer cannot fill — we fall through to [`Self::ocr_text`],
    /// preserving the previous OCR behavior for scanned PDFs.
    fn recognize_text(
        &self,
        pixel_bbox: &BBox,
        point_bbox: &BBox,
        image: &RgbImage,
        scale: f32,
        page_text: &PageText,
    ) -> RegionContent {
        if page_text.supports_native_fill() {
            if let [Some(filled)] = page_text.fill_regions(&[*point_bbox]).as_slice() {
                if !filled.text.is_empty() {
                    // Native text fills the whole region as one line; score 1.0
                    // marks it as exact (not a model confidence).
                    return RegionContent {
                        text_lines: vec![RecognizedLine {
                            bbox: *point_bbox,
                            text: filled.text.clone(),
                            score: 1.0,
                        }],
                        ..Default::default()
                    };
                }
            }
        }
        self.ocr_text(pixel_bbox, image, scale)
    }

    /// OCR fallback: detect text lines in the region crop, recognize each, returning
    /// lines in page-point coordinates. Missing det/rec models yield no lines. Used
    /// for scanned pages and any region the native text layer could not fill.
    fn ocr_text(&self, pixel_bbox: &BBox, image: &RgbImage, scale: f32) -> RegionContent {
        let (Some(det), Some(rec)) = (&self.models.ocr_det, &self.models.ocr_rec) else {
            return RegionContent::default();
        };
        let Some((crop, ox, oy)) = crop_region(image, pixel_bbox) else {
            return RegionContent::default();
        };

        let line_boxes = det.detect(&crop).unwrap_or_default();
        let mut text_lines = Vec::with_capacity(line_boxes.len());
        for line in line_boxes {
            // `line` is crop-local pixels; crop it out and recognize its text.
            let Some((line_crop, _, _)) = crop_region(&crop, &line) else {
                continue;
            };
            let (text, score) = match rec.recognize(&line_crop) {
                Ok(pair) => pair,
                Err(_) => continue,
            };
            // Map the crop-local line box back to page points: + crop origin (ox,
            // oy) to reach page pixels, then / scale to reach points.
            let page_box = scale_bbox(offset_bbox(&line, ox, oy), 1.0 / scale);
            text_lines.push(RecognizedLine {
                bbox: page_box,
                text,
                score,
            });
        }
        RegionContent {
            text_lines,
            ..Default::default()
        }
    }

    /// Formula: recognize LaTeX for the display-formula crop.
    fn recognize_formula(&self, pixel_bbox: &BBox, image: &RgbImage) -> RegionContent {
        let Some(model) = &self.models.formula else {
            return RegionContent::default();
        };
        let Some((crop, _, _)) = crop_region(image, pixel_bbox) else {
            return RegionContent::default();
        };
        RegionContent {
            latex: model.predict(&crop).ok(),
            ..Default::default()
        }
    }

    /// Table: classify wired/wireless and recognize into HTML.
    ///
    /// OCR spans are needed by the recognizers; this v1 passes an empty span set
    /// (structure-only) — full OCR-span matching is a later phase. Any missing
    /// model leaves `table_html` empty.
    fn recognize_table(&self, pixel_bbox: &BBox, image: &RgbImage) -> RegionContent {
        let Some((crop, _, _)) = crop_region(image, pixel_bbox) else {
            return RegionContent::default();
        };
        let html = match mineru_table::classify(&crop) {
            Ok(cls) => self.recognize_table_class(cls.class, &crop),
            // No classifier (model unavailable): try wireless as a default.
            Err(_) => self.recognize_wireless(&crop),
        };
        RegionContent {
            table_html: html,
            ..Default::default()
        }
    }

    /// Dispatches to the wired/wireless recognizer for a classified table.
    fn recognize_table_class(
        &self,
        class: mineru_table::TableClass,
        crop: &RgbImage,
    ) -> Option<mineru_types::Html> {
        match class {
            mineru_table::TableClass::Wireless => self.recognize_wireless(crop),
            mineru_table::TableClass::Wired => self
                .models
                .table_wired
                .as_ref()
                .and_then(|m| mineru_table::recognize_wired(m, crop, &[]).ok()),
        }
    }

    /// Wireless-table recognition, guarded on the loaded SLANet model.
    fn recognize_wireless(&self, crop: &RgbImage) -> Option<mineru_types::Html> {
        self.models
            .table_wireless
            .as_ref()
            .and_then(|m| mineru_table::recognize_wireless(m, crop, &[]).ok())
    }

    /// Image/chart: record a stable [`ImageRef`] for the region.
    ///
    /// Persisting the crop bytes is the caller's job (the writer stage owns the
    /// output image directory); here we only mint the reference so the assembled
    /// [`Document`] points at a deterministic path. The reference is the bare
    /// file name — the renderer joins it under the image directory (mirroring
    /// Python's `f"{img_bucket}/{image}"`), so baking a directory prefix in here
    /// would double it (`images/images/…`).
    fn crop_image(&self, page: usize, det: &LayoutDet, _image: &RgbImage) -> RegionContent {
        let name = format!("p{page}_o{}.png", det.order);
        RegionContent {
            image: Some(ImageRef(name)),
            ..Default::default()
        }
    }
}

#[async_trait]
impl Backend for PipelineBackend {
    async fn analyze(
        &self,
        input: DocInput,
        opts: &ParseOptions,
    ) -> Result<Document, BackendError> {
        self.run(&input, opts).map_err(Into::into)
    }
}

/// Resolves the `[start, end)` page range from options, clamped to the document.
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
/// in the source image's pixel space. `None` when the box has no area or lies
/// outside the image.
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

/// Translates a box by a pixel offset (used to map crop-local boxes back).
fn offset_bbox(bbox: &BBox, dx: f32, dy: f32) -> BBox {
    BBox::new(
        bbox.x0 + dx,
        bbox.y0 + dy,
        bbox.x1 + dx,
        bbox.y1 + dy,
    )
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
        let opts = ParseOptions::default();
        assert_eq!(page_bounds(5, &opts), (0, 5));
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
    fn crop_region_clamps_and_offsets() {
        let img = RgbImage::new(100, 100);
        let (crop, ox, oy) = crop_region(&img, &BBox::new(-10.0, 50.0, 40.0, 90.0)).unwrap();
        assert_eq!((ox, oy), (0.0, 50.0));
        assert_eq!(crop.dimensions(), (40, 40));
    }

    #[test]
    fn offset_translates_box() {
        let b = offset_bbox(&BBox::new(0.0, 0.0, 10.0, 10.0), 5.0, 7.0);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (5.0, 7.0, 15.0, 17.0));
    }
}
