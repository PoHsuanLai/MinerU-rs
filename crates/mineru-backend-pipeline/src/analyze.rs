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

use std::cell::Cell;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use burn::prelude::Backend as BurnBackend;
use image::{imageops::crop_imm, RgbImage};

use mineru_burn_common::backend::Cpu;
use mineru_layout::LayoutDet;
use mineru_pdf::{PageText, PdfiumLibrary, RenderOptions};
use mineru_table::{orientation::Rotation, OcrSpan};
use mineru_types::{
    Backend, BackendError, BBox, DocInput, Document, ImageRef, ImageWriter, Page, PageSize,
    ParseOptions,
};

use crate::assemble::{PageAssembler, RecognizedLine, Region, RegionContent, RegionKind};
use crate::models::PipelineModels;
use crate::para::merge_paragraphs;

/// Per-stage wall-clock accumulators, gated behind `MINERU_PROFILE=1`.
///
/// Purely diagnostic: when disabled every timing helper is a no-op and the
/// pipeline output is unchanged. The serial page loop means a plain [`Cell`]
/// (single-threaded, panic-free) suffices — no locking. Times are summed across
/// all pages and logged once at the end of [`PipelineBackend::run`].
#[derive(Default)]
struct Profile {
    enabled: bool,
    rasterize: Cell<Duration>,
    native_text: Cell<Duration>,
    layout: Cell<Duration>,
    ocr: Cell<Duration>,
    formula: Cell<Duration>,
    /// Orientation detection over table crops. Cheap on the common (upright)
    /// path, and two extra det+rec passes on the rare rotated one.
    table_orient: Cell<Duration>,
    table_classify: Cell<Duration>,
    /// OCR over table crops, kept separate from `ocr` (which covers text regions)
    /// so the table stages' true cost is visible.
    table_ocr: Cell<Duration>,
    table_recognize: Cell<Duration>,
}

impl Profile {
    /// Reads `MINERU_PROFILE`; enabled for any non-empty value other than `0`.
    fn from_env() -> Self {
        let enabled = std::env::var("MINERU_PROFILE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        Self {
            enabled,
            ..Default::default()
        }
    }

    /// Adds `elapsed` to `slot` (no-op when profiling is disabled).
    fn add(&self, slot: &Cell<Duration>, elapsed: Duration) {
        if self.enabled {
            slot.set(slot.get() + elapsed);
        }
    }

    /// Emits the accumulated per-stage summary as a single `tracing::info!` line.
    fn log_summary(&self) {
        if !self.enabled {
            return;
        }
        let ras = self.rasterize.get();
        let nat = self.native_text.get();
        let lay = self.layout.get();
        let ocr = self.ocr.get();
        let formula = self.formula.get();
        let t_ori = self.table_orient.get();
        let t_cls = self.table_classify.get();
        let t_ocr = self.table_ocr.get();
        let t_rec = self.table_recognize.get();
        let table = t_ori + t_cls + t_ocr + t_rec;
        let total = ras + nat + lay + ocr + formula + table;
        let pct = |d: Duration| {
            if total.is_zero() {
                0.0
            } else {
                d.as_secs_f64() / total.as_secs_f64() * 100.0
            }
        };
        tracing::info!(
            target: "mineru_profile",
            rasterize_s = ras.as_secs_f64(),
            rasterize_pct = pct(ras),
            native_text_s = nat.as_secs_f64(),
            native_text_pct = pct(nat),
            layout_s = lay.as_secs_f64(),
            layout_pct = pct(lay),
            ocr_s = ocr.as_secs_f64(),
            ocr_pct = pct(ocr),
            formula_s = formula.as_secs_f64(),
            formula_pct = pct(formula),
            table_orient_s = t_ori.as_secs_f64(),
            table_classify_s = t_cls.as_secs_f64(),
            table_ocr_s = t_ocr.as_secs_f64(),
            table_recognize_s = t_rec.as_secs_f64(),
            table_s = table.as_secs_f64(),
            table_pct = pct(table),
            stage_sum_s = total.as_secs_f64(),
            "per-stage wall-clock summary (sum of instrumented stages)"
        );
    }
}

/// The local Burn-model pipeline backend.
///
/// Owns the loaded [`PipelineModels`] and implements
/// [`Backend`](mineru_types::Backend). Cheap to share behind an `Arc`; the models
/// are immutable after construction.
///
/// Generic over the Burn backend `B` (default [`Cpu`]) of the neural stages; the
/// table stages inside [`PipelineModels`] always run on CPU (see its docs). Select
/// the GPU with `PipelineBackend::<mineru_burn_common::Gpu>::new(models)`.
pub struct PipelineBackend<B: BurnBackend = Cpu> {
    models: PipelineModels<B>,
    dpi: f32,
}

impl<B: BurnBackend> PipelineBackend<B> {
    /// Builds a backend from already-loaded models, rasterizing at the MinerU
    /// default 200 DPI.
    pub fn new(models: PipelineModels<B>) -> Self {
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

        let profile = Profile::from_env();

        // SERIAL iteration: PDFium is not safe for concurrent page ops.
        for index in start..end {
            let size = doc.page_size(index)?;
            let t = Instant::now();
            let rendered = doc.render_page(index, &render)?;
            profile.add(&profile.rasterize, t.elapsed());
            let image = rendered.into_inner();
            // Native text layer for this page (empty for scanned pages). A read
            // failure is non-fatal: fall back to OCR for the whole page.
            let t = Instant::now();
            let page_text = doc.extract_text(index).unwrap_or_else(|e| {
                tracing::warn!(page = index, error = %e, "native text extraction failed");
                PageText::default()
            });
            profile.add(&profile.native_text, t.elapsed());
            let page = self.analyze_page(
                index,
                size,
                &image,
                &page_text,
                opts.image_sink.as_deref(),
                &profile,
            );
            pages.push(page);
        }

        profile.log_summary();

        Ok(Document { pages })
    }

    /// Runs layout + recognition + assembly for one already-rasterized page.
    fn analyze_page(
        &self,
        index: usize,
        size: PageSize,
        image: &RgbImage,
        page_text: &PageText,
        sink: Option<&dyn ImageWriter>,
        profile: &Profile,
    ) -> Page {
        let scale = self.scale();

        // Layout is the driver; with no layout model the page is empty.
        let t = Instant::now();
        let dets = match &self.models.layout {
            Some(layout) => layout.detect(image).unwrap_or_else(|e| {
                tracing::warn!(page = index, error = %e, "layout detect failed");
                Vec::new()
            }),
            None => Vec::new(),
        };
        profile.add(&profile.layout, t.elapsed());

        let mut regions: Vec<Region> = dets
            .into_iter()
            .map(|det| self.recognize_region(index, det, image, scale, page_text, sink, profile))
            .collect();

        let t = Instant::now();
        self.fill_formulas(&mut regions, image, scale);
        profile.add(&profile.formula, t.elapsed());

        // Tables run last: a formula printed inside one is detected as a page
        // formula, so its LaTeX only exists once `fill_formulas` has run, and the
        // table needs it to fill the cell the formula sits in.
        let claimed = self.fill_tables(index, &mut regions, image, scale, sink, profile);
        drop_regions(&mut regions, &claimed);

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
    #[allow(clippy::too_many_arguments)]
    fn recognize_region(
        &self,
        page: usize,
        det: LayoutDet,
        image: &RgbImage,
        scale: f32,
        page_text: &PageText,
        sink: Option<&dyn ImageWriter>,
        profile: &Profile,
    ) -> Region {
        let pixel_bbox = det.bbox;
        // Detection box in page points (the space native text lives in).
        let point_bbox = scale_bbox(pixel_bbox, 1.0 / scale);
        let content = match RegionKind::classify(det.label) {
            RegionKind::Text(_) | RegionKind::Caption | RegionKind::Footnote => {
                let t = Instant::now();
                let c = self.recognize_text(&pixel_bbox, &point_bbox, image, scale, page_text);
                profile.add(&profile.ocr, t.elapsed());
                c
            }
            RegionKind::Discarded(_) => {
                let t = Instant::now();
                let c = self.recognize_text(&pixel_bbox, &point_bbox, image, scale, page_text);
                profile.add(&profile.ocr, t.elapsed());
                c
            }
            // Formulas are left empty here and filled by `fill_formulas` after the
            // whole page is mapped, so every crop on the page decodes in one batched
            // pass instead of one model call each.
            RegionKind::Equation | RegionKind::InlineFormula => RegionContent::default(),
            // Filled by `fill_tables` once formulas are recognized; a table may
            // need their LaTeX for its own cells.
            RegionKind::Table => RegionContent::default(),
            RegionKind::Image | RegionKind::Chart => {
                self.crop_image(page, &det, &pixel_bbox, image, sink)
            }
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

    /// Recognizes every formula on the page in one batched decode and fills the
    /// results into their regions.
    ///
    /// Display and inline formulas go through the same model and the same batch —
    /// as they do in the reference, which feeds both labels to one MFR call — and
    /// differ only in which [`RegionContent`] field receives the LaTeX. The
    /// assembler folds `inline_latex` into the surrounding text as `$…$` and
    /// `latex` into a standalone `$$…$$` block.
    ///
    /// Batching is what makes this worth the two-pass shape: the decoder is
    /// autoregressive, so at batch 1 its matmuls are matrix-vector and the model
    /// weights are re-read for every crop. One call per page amortizes that read
    /// across the page's formulas.
    ///
    /// A crop that fails to preprocess yields `None` for that lane alone and leaves
    /// its region empty, which is what the per-crop path did with `.ok()`.
    fn fill_formulas(&self, regions: &mut [Region], image: &RgbImage, scale: f32) {
        let Some(model) = &self.models.formula else {
            return;
        };

        // Region bboxes are in page points by now; the crop has to come from the
        // pixel raster, so undo the scaling `recognize_region` applied.
        let mut lanes: Vec<(usize, RgbImage)> = Vec::new();
        for (i, region) in regions.iter().enumerate() {
            match RegionKind::classify(region.det.label) {
                RegionKind::Equation | RegionKind::InlineFormula => {}
                _ => continue,
            }
            let pixel_bbox = scale_bbox(region.det.bbox, scale);
            if let Some((crop, _, _)) = crop_region(image, &pixel_bbox) {
                lanes.push((i, crop));
            }
        }
        if lanes.is_empty() {
            return;
        }

        let crops: Vec<RgbImage> = lanes.iter().map(|(_, crop)| crop.clone()).collect();
        let results = match model.predict_batch(&crops) {
            Ok(results) => results,
            Err(e) => {
                tracing::warn!(error = %e, "batched formula decode failed; page has no formulas");
                return;
            }
        };
        if results.len() != lanes.len() {
            tracing::warn!(
                got = results.len(),
                want = lanes.len(),
                "batched formula decode returned the wrong lane count; page has no formulas"
            );
            return;
        }

        for ((region_index, _), latex) in lanes.iter().zip(results) {
            let Some(latex) = latex else { continue };
            let Some(region) = regions.get_mut(*region_index) else {
                continue;
            };
            match RegionKind::classify(region.det.label) {
                RegionKind::InlineFormula => region.content.inline_latex = Some(latex),
                _ => region.content.latex = Some(latex),
            }
        }
    }

    /// Recognizes every table on the page, routing the page's formulas into the
    /// tables that contain them, and returns the region indices whose formulas were
    /// consumed by a table.
    ///
    /// A formula printed inside a table is never seen by the table path: the page
    /// layout model detects it as an ordinary page formula, and [`fill_formulas`]
    /// has already turned it into LaTeX by the time this runs. Left alone, that
    /// LaTeX is emitted as a block beside the table while the table's own cell
    /// holds whatever OCR made of the formula's glyphs. So each table claims the
    /// formulas whose centers fall inside it, masks their pixels out of its crop,
    /// and hands their LaTeX to the structure matcher as ordinary spans — the
    /// reference's design (`_extract_table_inline_objects`), and the reason
    /// [`mineru_table::inline`] needs no formula model of its own.
    ///
    /// [`fill_formulas`]: Self::fill_formulas
    fn fill_tables(
        &self,
        page: usize,
        regions: &mut [Region],
        image: &RgbImage,
        scale: f32,
        sink: Option<&dyn ImageWriter>,
        profile: &Profile,
    ) -> Vec<usize> {
        use mineru_table::inline::PageFormula;

        // Region bboxes are page points by now; formulas and table boxes are
        // compared in the pixel space the crops come from.
        let mut formulas: Vec<PageFormula> = Vec::new();
        let mut formula_regions: Vec<usize> = Vec::new();
        for (i, region) in regions.iter().enumerate() {
            let latex = match RegionKind::classify(region.det.label) {
                RegionKind::Equation => region.content.latex.as_ref(),
                RegionKind::InlineFormula => region.content.inline_latex.as_ref(),
                _ => None,
            };
            if let Some(latex) = latex {
                formulas.push(PageFormula {
                    bbox: scale_bbox(region.det.bbox, scale),
                    latex: latex.0.clone(),
                });
                formula_regions.push(i);
            }
        }

        // The three stay index-aligned: a table that cannot be cropped is skipped
        // in all of them, and `table_boxes` is what `assign_to_tables` indexes.
        let mut table_regions: Vec<usize> = Vec::new();
        let mut table_boxes: Vec<Option<BBox>> = Vec::new();
        let mut crops: Vec<RgbImage> = Vec::new();
        for (i, region) in regions.iter().enumerate() {
            if !matches!(RegionKind::classify(region.det.label), RegionKind::Table) {
                continue;
            }
            let pixel_bbox = scale_bbox(region.det.bbox, scale);
            let Some((crop, _, _)) = crop_region(image, &pixel_bbox) else {
                continue;
            };
            let t = Instant::now();
            let (crop, rotation) = self.deskew_table(crop);
            profile.add(&profile.table_orient, t.elapsed());

            table_regions.push(i);
            // A rotated crop has moved its pixels out from under the page-space
            // box, so this table opts out of formulas (as the reference does).
            table_boxes.push((rotation == Rotation::None).then_some(pixel_bbox));
            crops.push(crop);
        }

        let assignment = mineru_table::assign_to_tables(&table_boxes, &formulas);

        for (slot, (&region_index, crop)) in table_regions.iter().zip(crops).enumerate() {
            // Resolve each assigned formula to the span text and box the matcher
            // needs, so the table path never sees the formula types at all.
            let inline: Vec<OcrSpan> = assignment
                .per_table
                .get(slot)
                .map_or(&[][..], |v| &v[..])
                .iter()
                .filter_map(|f| {
                    let latex = &formulas.get(f.formula)?.latex;
                    // Delimiters go on here rather than at render time: the matcher
                    // splices span text into cells verbatim, and by the time the
                    // HTML exists a formula cell is indistinguishable from a text
                    // one. Score 1.0 marks it exact — a model output, not an OCR
                    // read.
                    Some(OcrSpan::new(f.crop_bbox, format!("${latex}$"), 1.0))
                })
                .collect();
            let order = match regions.get(region_index) {
                Some(region) => region.det.order,
                None => continue,
            };
            let content = self.recognize_table(page, order, crop, &inline, sink, profile);
            if let Some(region) = regions.get_mut(region_index) {
                region.content = content;
            }
        }

        // Map claimed formula indices back to the regions they came from.
        assignment
            .claimed
            .iter()
            .filter_map(|&f| formula_regions.get(f).copied())
            .collect()
    }

    /// Table: classify wired/wireless and recognize into HTML.
    ///
    /// `crop` arrives already deskewed (see [`deskew_table`](Self::deskew_table)),
    /// so the classifier, structure model and cell OCR all see an upright table.
    /// Any missing model leaves `table_html` empty.
    ///
    /// `inline` carries the spans for formulas printed inside this table, in
    /// crop-local pixels, already carrying their delimiters —
    /// [`fill_tables`](Self::fill_tables) resolves those and is the only caller
    /// that has the page context to do so. They are masked out of the crop before
    /// OCR and appended afterwards; see the masking note below.
    fn recognize_table(
        &self,
        page: usize,
        order: usize,
        crop: RgbImage,
        inline: &[OcrSpan],
        sink: Option<&dyn ImageWriter>,
        profile: &Profile,
    ) -> RegionContent {
        let t = Instant::now();
        // Tables are wired on CPU (see `PipelineModels`); classify on the same
        // backend as the recognizers it dispatches to.
        let classified = mineru_table::classify::<Cpu>(&crop);
        profile.add(&profile.table_classify, t.elapsed());
        // Cell text comes from OCR over the crop; without it both recognizers
        // return their predicted grid with every cell empty.
        let t = Instant::now();
        // OCR reads a formula's glyphs as ordinary text, which would compete with
        // its LaTeX for the same cell, so the formulas' pixels are painted out
        // first and their LaTeX added back as spans below.
        let spans = if inline.is_empty() {
            self.table_spans(&crop)
        } else {
            let masked = mineru_table::mask_boxes(&crop, inline.iter().map(|s| s.bbox));
            let mut spans = self.table_spans(&masked);
            spans.extend_from_slice(inline);
            spans
        };
        profile.add(&profile.table_ocr, t.elapsed());
        let t = Instant::now();
        let html = match classified {
            Ok(cls) => self.recognize_classified(&cls, &crop, &spans),
            // No classifier (model unavailable): try wireless as a default.
            Err(_) => self.recognize_wireless(&crop, &spans),
        };
        profile.add(&profile.table_recognize, t.elapsed());
        // Persist the table crop (best-effort) and mint its ref only when a sink is
        // present, mirroring `crop_image`; with no sink the image ref stays `None`
        // (unchanged behavior). The assembler forwards `content.image` into
        // `TableBody.image`.
        let image = sink.map(|sink| {
            let name = format!("p{page}_o{order}.png");
            write_crop(sink, page, order, &name, &crop);
            ImageRef(name)
        });
        RegionContent {
            table_html: html,
            image,
            ..Default::default()
        }
    }

    /// Recognizes a classified table, running *both* engines when the
    /// classification is not decisive and keeping the better result.
    ///
    /// The classifier judges a 224x224 view of the whole crop, which is a weak
    /// signal for the thing that actually matters here: whether the wired engine
    /// can find the rules. A table with faint or partial ruling reads as
    /// borderless to it while the wired engine still recovers the grid, and a
    /// confidently-wireless call can still be wrong. So a `Wired` call — or a
    /// `Wireless` one under
    /// [`WIRELESS_TRUST_THRESHOLD`](mineru_table::WIRELESS_TRUST_THRESHOLD) —
    /// runs both and lets [`mineru_table::select`] compare what they produced,
    /// mirroring `batch_analyze.py:666-670` + `unet_table/main.py:337-357`.
    ///
    /// Only a confidently-wireless table skips the wired engine, which is the
    /// cost side of this: everything else pays a UNet forward.
    ///
    /// `spans` are the crop-local OCR detections both recognizers match onto the
    /// structure they predict; see [`table_spans`](Self::table_spans).
    fn recognize_classified(
        &self,
        cls: &mineru_table::Classification,
        crop: &RgbImage,
        spans: &[OcrSpan],
    ) -> Option<mineru_types::Html> {
        use mineru_table::{Choice, TableClass, WIRELESS_TRUST_THRESHOLD};

        let confident_wireless =
            cls.class == TableClass::Wireless && cls.score >= WIRELESS_TRUST_THRESHOLD;
        if confident_wireless {
            return self.recognize_wireless(crop, spans);
        }

        // Both engines, then decide on their output rather than on the score.
        let wireless = self.recognize_wireless(crop, spans);
        let wired = self.recognize_wired(crop, spans);
        match (wired, wireless) {
            (Some(wired), Some(wireless)) => {
                let choice = mineru_table::select(&wired.0, &wireless.0, spans);
                tracing::debug!(
                    ?choice,
                    class = ?cls.class,
                    score = cls.score,
                    wired_cells = wired.0.matches("<td").count() + wired.0.matches("<th").count(),
                    wireless_cells = wireless.0.matches("<td").count() + wireless.0.matches("<th").count(),
                    "picked a table engine by comparing both recognitions"
                );
                Some(match choice {
                    Choice::Wired => wired,
                    Choice::Wireless => wireless,
                })
            }
            // One engine failed or is unavailable: the other is the only answer.
            (wired, wireless) => wired.or(wireless),
        }
    }

    /// Wired-table recognition, guarded on the loaded UNet model.
    fn recognize_wired(&self, crop: &RgbImage, spans: &[OcrSpan]) -> Option<mineru_types::Html> {
        self.models
            .table_wired
            .as_ref()
            .and_then(|m| warn_on_err("wired", mineru_table::recognize_wired(m, crop, spans)))
    }

    /// Wireless-table recognition, guarded on the loaded SLANet model.
    fn recognize_wireless(&self, crop: &RgbImage, spans: &[OcrSpan]) -> Option<mineru_types::Html> {
        self.models.table_wireless.as_ref().and_then(|m| {
            warn_on_err("wireless", mineru_table::recognize_wireless(m, crop, spans))
        })
    }

    /// Rotates a sideways-typeset table crop upright, returning it unchanged when
    /// it already is (the overwhelmingly common case).
    ///
    /// The chosen [`Rotation`] is returned alongside the crop because it invalidates
    /// page-space geometry: anything computed against the unrotated table box (the
    /// inline formulas of [`fill_table_formulas`](Self::fill_table_formulas)) no
    /// longer lines up once the pixels move.
    ///
    /// Wide tables are often printed rotated to fit a portrait page. OCR reads
    /// such a crop as confident nonsense rather than failing — `Forest age
    /// (years)` becomes `20 20 20 20 20` — so nothing downstream can detect it.
    ///
    /// The policy lives in [`mineru_table::orientation`]; this supplies the OCR
    /// it decides on. A cheap detection-box gate runs first, because scoring
    /// costs two extra det+rec passes and almost no tables need it.
    ///
    /// Without a det/rec model there is nothing to score with, so the crop is
    /// returned as-is.
    fn deskew_table(&self, crop: RgbImage) -> (RgbImage, Rotation) {
        use mineru_table::orientation::{
            is_rotation_candidate, select_rotation, OrientationScore, Rotation,
        };

        let (Some(det), Some(_)) = (&self.models.ocr_det, &self.models.ocr_rec) else {
            return (crop, Rotation::None);
        };

        let boxes = det.detect(&crop).unwrap_or_default();
        if !is_rotation_candidate(&boxes) {
            return (crop, Rotation::None);
        }

        let scores: Vec<(Rotation, OrientationScore)> = Rotation::ALL
            .iter()
            .map(|&rotation| {
                let view = rotate(&crop, rotation);
                // Re-detect per angle: box shapes are what changed, and the 0°
                // boxes do not carry over to a rotated view.
                let view_boxes = det.detect(&view).unwrap_or_default();
                (rotation, self.score_orientation(&view, &view_boxes))
            })
            .collect();

        let choice = select_rotation(&scores);
        if choice != Rotation::None {
            tracing::debug!(
                rotation = ?choice,
                scores = ?scores.iter().map(|(r, s)| (r, s.confidence)).collect::<Vec<_>>(),
                "rotated a sideways table crop upright"
            );
        }
        (rotate(&crop, choice), choice)
    }

    /// Scores how well OCR reads `view`, over an even sample of its boxes.
    fn score_orientation(
        &self,
        view: &RgbImage,
        boxes: &[BBox],
    ) -> mineru_table::orientation::OrientationScore {
        use mineru_table::orientation::OrientationScore;

        let Some(rec) = &self.models.ocr_rec else {
            return OrientationScore::ZERO;
        };
        let reads: Vec<(String, f32)> = mineru_table::orientation::sample_boxes(boxes)
            .into_iter()
            .filter_map(|b| {
                let (line_crop, _, _) = crop_region(view, &b)?;
                rec.recognize(&line_crop).ok()
            })
            .collect();
        OrientationScore::from_reads(reads.iter().map(|(t, s)| (t.as_str(), *s)))
    }

    /// OCRs a table crop into the spans the structure matcher fills cells from.
    ///
    /// Both table recognizers predict a cell grid and then assign each OCR span to
    /// a cell by IoU against the predicted cell boxes, so the spans **must** be in
    /// the same space those boxes are: crop-local pixels. That is what the
    /// detector already returns here, so — unlike [`ocr_text`](Self::ocr_text),
    /// which maps lines back to page points for the assembled `Document` — no
    /// coordinate mapping is applied.
    ///
    /// Missing det/rec models yield no spans, which degrades a table to its
    /// structure with empty cells rather than failing the region.
    fn table_spans(&self, crop: &RgbImage) -> Vec<OcrSpan> {
        let (Some(det), Some(rec)) = (&self.models.ocr_det, &self.models.ocr_rec) else {
            return Vec::new();
        };
        det.detect(crop)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|line| {
                let (line_crop, _, _) = crop_region(crop, &line)?;
                let (text, score) = rec.recognize(&line_crop).ok()?;
                Some(OcrSpan::new(line, text, score))
            })
            .collect()
    }

    /// Image/chart: record a stable [`ImageRef`] for the region.
    ///
    /// Persisting the crop bytes is the caller's job (the writer stage owns the
    /// output image directory); here we only mint the reference so the assembled
    /// [`Document`] points at a deterministic path. The reference is the bare
    /// file name — the renderer joins it under the image directory (mirroring
    /// Python's `f"{img_bucket}/{image}"`), so baking a directory prefix in here
    /// would double it (`images/images/…`).
    fn crop_image(
        &self,
        page: usize,
        det: &LayoutDet,
        pixel_bbox: &BBox,
        image: &RgbImage,
        sink: Option<&dyn ImageWriter>,
    ) -> RegionContent {
        // Image/chart regions always mint the ref (unchanged behavior). When a
        // sink is present, also crop the pixel bbox and persist it (best-effort);
        // the minted name matches what is written.
        let name = format!("p{page}_o{}.png", det.order);
        if let Some(sink) = sink {
            if let Some((crop, _, _)) = crop_region(image, pixel_bbox) {
                write_crop(sink, page, det.order, &name, &crop);
            }
        }
        RegionContent {
            image: Some(ImageRef(name)),
            ..Default::default()
        }
    }
}

/// Logs a table-recognition failure and degrades to "unrecognized".
///
/// A table failure must not abort the parse, but it must not vanish either: the
/// most common cause is the `.bpk` weight fetch failing (see
/// `mineru_table::weights::DEFAULT_WEIGHTS_BASE`), and swallowing that silently
/// leaves the user with tables missing from the output and no clue why.
fn warn_on_err(
    kind: &str,
    result: mineru_table::Result<mineru_types::Html>,
) -> Option<mineru_types::Html> {
    match result {
        Ok(html) => Some(html),
        Err(e) => {
            tracing::warn!(table = kind, error = %e, "table recognition failed; emitting no table");
            None
        }
    }
}

/// Removes the regions at `indices` (positions in `regions`), ignoring any that
/// are out of range.
///
/// Used to drop formulas a table absorbed into a cell, which would otherwise be
/// emitted a second time as a standalone block next to that table.
fn drop_regions(regions: &mut Vec<Region>, indices: &[usize]) {
    if indices.is_empty() {
        return;
    }
    let mut position = 0usize;
    regions.retain(|_| {
        let keep = !indices.contains(&position);
        position += 1;
        keep
    });
}

/// Writes an already-cropped region PNG through the sink, best-effort. A write
/// error is logged and swallowed: a failed crop must not abort the parse. `name`
/// is the ref name the caller mints, so the file written matches the reference.
fn write_crop(sink: &dyn ImageWriter, page: usize, order: usize, name: &str, crop: &RgbImage) {
    if let Err(e) = mineru_io::write_png(sink, name, crop) {
        tracing::warn!(page, order, error = %e, "failed to write region crop");
    }
}

#[async_trait]
impl<B: BurnBackend> Backend for PipelineBackend<B> {
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

/// Applies a [`Rotation`] to an image.
///
/// [`Rotation::None`] clones rather than borrowing so callers can hand back one
/// owned image whichever branch they take; a table crop is small and this runs
/// at most once per table.
///
/// [`Rotation`]: mineru_table::orientation::Rotation
fn rotate(image: &RgbImage, rotation: mineru_table::orientation::Rotation) -> RgbImage {
    use mineru_table::orientation::Rotation;
    match rotation {
        Rotation::None => image.clone(),
        // `rotate90` maps the left edge to the top; `rotate270` the right edge.
        Rotation::LeftEdgeToTop => image::imageops::rotate90(image),
        Rotation::RightEdgeToTop => image::imageops::rotate270(image),
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

    /// A region tagged by `order`, so a survivor can be identified after removal.
    fn region(order: usize) -> Region {
        Region {
            det: LayoutDet {
                bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
                label: mineru_layout::LayoutLabel::Text,
                score: 1.0,
                order,
            },
            content: RegionContent::default(),
        }
    }

    fn orders(regions: &[Region]) -> Vec<usize> {
        regions.iter().map(|r| r.det.order).collect()
    }

    /// Asserts on *which* regions survive, not how many: removing by shifting
    /// indices is exactly the bug an count-only check would miss.
    #[test]
    fn drop_regions_removes_only_the_named_positions() {
        let mut regions: Vec<Region> = (0..5).map(region).collect();
        drop_regions(&mut regions, &[1, 3]);
        assert_eq!(orders(&regions), vec![0, 2, 4]);
    }

    #[test]
    fn drop_regions_with_no_indices_keeps_everything() {
        let mut regions: Vec<Region> = (0..3).map(region).collect();
        drop_regions(&mut regions, &[]);
        assert_eq!(orders(&regions), vec![0, 1, 2]);
    }

    #[test]
    fn drop_regions_ignores_out_of_range_indices() {
        let mut regions: Vec<Region> = (0..2).map(region).collect();
        drop_regions(&mut regions, &[0, 99]);
        assert_eq!(orders(&regions), vec![1]);
    }

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
    fn rotation_maps_the_named_edge_to_the_top() {
        use mineru_table::orientation::Rotation;

        // A 2x1 image: left pixel red, right pixel blue. After a quarter turn the
        // named edge's pixel must be the one on top. Asserting on pixels rather
        // than on `rotate90`/`rotate270` names is the point: the direction
        // conventions invert between libraries, and getting this backwards
        // produces a 180°-wrong crop that OCR reads as plausible garbage.
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0])); // left
        img.put_pixel(1, 0, image::Rgb([0, 0, 255])); // right

        let left_up = rotate(&img, Rotation::LeftEdgeToTop);
        assert_eq!(left_up.dimensions(), (1, 2));
        assert_eq!(*left_up.get_pixel(0, 0), image::Rgb([255, 0, 0]));

        let right_up = rotate(&img, Rotation::RightEdgeToTop);
        assert_eq!(right_up.dimensions(), (1, 2));
        assert_eq!(*right_up.get_pixel(0, 0), image::Rgb([0, 0, 255]));

        let same = rotate(&img, Rotation::None);
        assert_eq!(same.dimensions(), (2, 1));
        assert_eq!(*same.get_pixel(0, 0), image::Rgb([255, 0, 0]));
    }

    #[test]
    fn offset_translates_box() {
        let b = offset_bbox(&BBox::new(0.0, 0.0, 10.0, 10.0), 5.0, 7.0);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (5.0, 7.0, 15.0, 17.0));
    }

    use std::sync::Mutex;

    use crate::models::PipelineModels;

    /// Test sink that records each `(name, byte_len)` it is handed.
    #[derive(Debug, Default)]
    struct RecordingSink {
        writes: Mutex<Vec<(String, usize)>>,
    }

    impl ImageWriter for RecordingSink {
        fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
            if let Ok(mut w) = self.writes.lock() {
                w.push((name.to_string(), bytes.len()));
            }
            Ok(())
        }
    }

    /// A gradient RgbImage so the encoded PNG has non-trivial content.
    fn synthetic_image(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        })
    }

    #[test]
    fn crop_image_writes_one_png_matching_ref() {
        let backend = PipelineBackend {
            models: PipelineModels::<Cpu>::default(),
            dpi: mineru_pdf::DEFAULT_DPI,
        };
        let image = synthetic_image(200, 200);
        let det = LayoutDet::new(
            BBox::new(10.0, 20.0, 110.0, 120.0),
            mineru_layout::LayoutLabel::Chart,
            0.9,
            7,
        );
        let sink = RecordingSink::default();

        let content = backend.crop_image(3, &det, &det.bbox, &image, Some(&sink));

        // Ref name is minted and matches the written file.
        let expected = format!("p3_o{}.png", det.order);
        assert_eq!(content.image, Some(ImageRef(expected.clone())));

        let writes = sink.writes.lock().unwrap();
        assert_eq!(writes.len(), 1, "exactly one crop should be written");
        assert_eq!(writes[0].0, expected, "written name must match the ref");
        assert!(writes[0].1 > 100, "PNG bytes should be non-trivial");
    }

    #[test]
    fn crop_image_without_sink_mints_ref_but_writes_nothing() {
        let backend = PipelineBackend {
            models: PipelineModels::<Cpu>::default(),
            dpi: mineru_pdf::DEFAULT_DPI,
        };
        let image = synthetic_image(50, 50);
        let det = LayoutDet::new(
            BBox::new(0.0, 0.0, 40.0, 40.0),
            mineru_layout::LayoutLabel::Chart,
            0.9,
            2,
        );

        let content = backend.crop_image(1, &det, &det.bbox, &image, None);
        assert_eq!(content.image, Some(ImageRef(format!("p1_o{}.png", det.order))));
    }
}
