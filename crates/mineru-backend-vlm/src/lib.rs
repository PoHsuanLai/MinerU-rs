//! VLM backend: the [`Backend`](mineru_types::Backend) implementation backed by an
//! external OpenAI-compatible VLM server.
//!
//! [`VlmBackend`] opens the input PDF, rasterizes each page (in the requested page
//! range) to an RGB image, runs the two-step MinerU2.5 extraction over each page
//! via [`mineru_vlm_client::VlmClient`], and assembles the per-page output into the
//! canonical [`Document`](mineru_types::Document) tree with
//! [`assemble_document`](mineru_vlm_client::assemble_document).
//!
//! # Threading
//!
//! PDFium (used for rasterization) keeps process-global native state and is **not**
//! safe for concurrent document use, so pages are rasterized **serially**. Each
//! page is rendered and then its VLM extraction awaited before moving on; no
//! concurrent rasterization is attempted. See [`mineru_pdf`] for the constraint.

pub mod error;

use async_trait::async_trait;

use mineru_pdf::{PdfiumLibrary, RenderOptions};
use mineru_types::{Backend, BackendError, DocInput, Document, ParseOptions};
use mineru_vlm_client::{assemble_document, VlmClient, VlmClientConfig, VlmPage};

pub use error::{Error, Result};

/// The VLM parsing backend.
///
/// Owns a configured [`VlmClient`]. Cheap to share behind an `Arc`; the client is
/// immutable after construction.
pub struct VlmBackend {
    client: VlmClient,
    dpi: f32,
}

impl VlmBackend {
    /// Builds a backend from VLM client configuration, rasterizing at the MinerU
    /// default 200 DPI.
    pub fn new(config: VlmClientConfig) -> Self {
        Self {
            client: VlmClient::new(config),
            dpi: mineru_pdf::DEFAULT_DPI,
        }
    }

    /// Overrides the rasterization DPI (default 200).
    pub fn with_dpi(mut self, dpi: f32) -> Self {
        self.dpi = dpi;
        self
    }

    /// Rasterizes and extracts every requested page, in order, then assembles the
    /// per-page output into a [`Document`].
    ///
    /// Rendering and extraction are interleaved **serially**: a page is rasterized
    /// and its extraction awaited before the next page is touched, honoring
    /// PDFium's single-threaded constraint (see [`mineru_pdf`]).
    async fn run(&self, input: &DocInput, opts: &ParseOptions) -> Result<Document> {
        let lib = PdfiumLibrary::load()?;
        let doc = lib.open(&input.bytes)?;
        let render = RenderOptions::with_dpi(self.dpi);

        let (start, end) = page_bounds(doc.page_count(), opts);

        // `image_analysis` enables content extraction for image/chart blocks; the
        // reference client gates this behind the same flags MinerU uses for
        // formula/table recognition, so we enable it when either is requested.
        let image_analysis = opts.formula || opts.table;

        let mut pages: Vec<VlmPage> = Vec::with_capacity(end.saturating_sub(start));
        for index in start..end {
            // Render this page (serial: PDFium is not concurrency-safe), then
            // await its extraction before rendering the next.
            let image = doc.render_page(index, &render)?.into_inner();
            let page = self.client.extract_page(&image, image_analysis).await?;
            pages.push(page);
        }

        Ok(assemble_document(pages))
    }
}

#[async_trait]
impl Backend for VlmBackend {
    async fn analyze(
        &self,
        input: DocInput,
        opts: &ParseOptions,
    ) -> std::result::Result<Document, BackendError> {
        self.run(&input, opts).await.map_err(Into::into)
    }
}

/// Resolves the `[start, end)` page range from options, clamped to the document.
///
/// Mirrors the pipeline backend's bounds logic: `page_range` is an inclusive start
/// and an optional exclusive end, both clamped into `0..=page_count`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_from_default_config() {
        // Construction must not require a live server; it only wires the client.
        let backend = VlmBackend::new(VlmClientConfig::default());
        assert_eq!(backend.dpi, mineru_pdf::DEFAULT_DPI);
    }

    #[test]
    fn with_dpi_overrides() {
        let backend = VlmBackend::new(VlmClientConfig::default()).with_dpi(144.0);
        assert_eq!(backend.dpi, 144.0);
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

    /// A real analyze against a live VLM server + a demo PDF. Ignored by default
    /// because it needs an external server; run with:
    ///
    /// ```text
    /// MINERU_PDFIUM_LIB_PATH=/path/to/libpdfium.dylib \
    ///   cargo test -p mineru-backend-vlm -- --ignored
    /// ```
    #[tokio::test]
    #[ignore = "requires a live VLM server and a matching libpdfium native library"]
    async fn analyzes_demo_pdf() {
        let demo_dir =
            std::env::var("MINERU_DEMO_DIR").expect("set MINERU_DEMO_DIR to the demo/pdfs directory");
        let bytes = std::fs::read(std::path::Path::new(&demo_dir).join("demo1.pdf"))
            .expect("demo pdf present");
        let backend = VlmBackend::new(VlmClientConfig::default());
        let doc = backend
            .analyze(DocInput::new(bytes), &ParseOptions::default())
            .await
            .expect("analyze demo pdf");
        assert!(!doc.pages.is_empty());
    }
}
