//! PDF input for MinerU.
//!
//! Opens PDF bytes, reports page metadata, and rasterizes pages to RGB images via
//! the PDFium native library (bound at runtime). Native-text extraction (the
//! `pdftext`-style span grouping) is stubbed and lands in a later phase.
//!
//! # PDFium library resolution
//!
//! PDFium is loaded dynamically at runtime. [`PdfiumLibrary::load`] searches, in
//! order: the `MINERU_PDFIUM_LIB_PATH` environment variable, a set of common
//! system locations, and finally the platform's default lookup. No native library
//! is bundled, keeping the crate small.
//!
//! # Threading
//!
//! PDFium keeps process-global native state and is **not safe for concurrent use**
//! across threads, even through a single binding. [`PdfiumLibrary::load`] returns a
//! shared, bind-once instance, but callers must serialize actual document
//! operations — parse one document at a time, or guard access with a mutex. The
//! higher layers process documents sequentially for exactly this reason.

pub mod error;

use std::path::PathBuf;
use std::sync::OnceLock;

use image::RgbImage;
use pdfium_render::prelude::{PdfRenderConfig, Pdfium};

pub use error::{Error, Result};
use mineru_types::PageSize;

/// Process-global PDFium binding, bound lazily exactly once.
///
/// PDFium initializes global native state on load and must not be bound more than
/// once per process (nor dropped while in use), so the result of the single bind
/// attempt — success or the error message — is cached here and shared.
static LIBRARY: OnceLock<std::result::Result<PdfiumLibrary, String>> = OnceLock::new();

/// Default rasterization resolution, matching the Python pipeline.
pub const DEFAULT_DPI: f32 = 200.0;

/// Options controlling how a page is rasterized.
#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    /// Target resolution in dots per inch.
    pub dpi: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self { dpi: DEFAULT_DPI }
    }
}

impl RenderOptions {
    /// Builds render options at the given DPI.
    pub fn with_dpi(dpi: f32) -> Self {
        Self { dpi }
    }

    /// Pixels per PDF point (1 point = 1/72 inch).
    fn scale(&self) -> f32 {
        self.dpi / 72.0
    }
}

/// A rasterized page: an RGB bitmap plus its pixel dimensions.
#[derive(Debug, Clone)]
pub struct PageImage {
    pub width: u32,
    pub height: u32,
    image: RgbImage,
}

impl PageImage {
    /// Borrows the underlying RGB image.
    pub fn as_rgb(&self) -> &RgbImage {
        &self.image
    }

    /// Consumes the wrapper, yielding the RGB image.
    pub fn into_inner(self) -> RgbImage {
        self.image
    }
}

/// Placeholder for a page's native (embedded) text.
///
/// Populated once the `pdftext`-style span-grouping is implemented.
#[derive(Debug, Clone, Default)]
pub struct PageText {
    pub spans: Vec<()>,
}

/// A handle to the loaded PDFium native library.
///
/// Owns the binding; [`PdfDocument`] borrows from it so the library outlives every
/// document opened against it.
pub struct PdfiumLibrary {
    pdfium: Pdfium,
}

impl PdfiumLibrary {
    /// Returns the process-global PDFium binding, loading it on first use.
    ///
    /// PDFium must be bound only once per process, so every caller shares one
    /// instance. Search order for the native library: `MINERU_PDFIUM_LIB_PATH`,
    /// common system paths, then the platform default.
    pub fn load() -> Result<&'static Self> {
        // Fast path: already bound.
        if let Some(slot) = LIBRARY.get() {
            return slot.as_ref().map_err(|e| Error::Bind(e.clone()));
        }
        // Slow path: bind exactly once under a lock so PDFium is never
        // initialized twice or dropped while another thread holds it. Two racing
        // binds would double-init and then destroy the loser mid-use → segfault.
        static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _init = INIT.lock().unwrap_or_else(|e| e.into_inner());
        let slot = LIBRARY.get_or_init(|| Self::bind().map_err(|e| e.to_string()));
        slot.as_ref().map_err(|e| Error::Bind(e.clone()))
    }

    /// Binds the native library from the first location that resolves.
    fn bind() -> Result<Self> {
        for candidate in candidate_library_paths() {
            if let Ok(bindings) = Pdfium::bind_to_library(&candidate) {
                return Ok(Self {
                    pdfium: Pdfium::new(bindings),
                });
            }
        }
        Pdfium::bind_to_system_library()
            .map(|bindings| Self {
                pdfium: Pdfium::new(bindings),
            })
            .map_err(|e| Error::Bind(e.to_string()))
    }

    /// Opens a PDF from in-memory bytes.
    ///
    /// The borrow ties the document to `bytes` (PDFium reads from the slice
    /// lazily) as well as to the library.
    pub fn open<'a>(&'a self, bytes: &'a [u8]) -> Result<PdfDocument<'a>> {
        let doc = self
            .pdfium
            .load_pdf_from_byte_slice(bytes, None)
            .map_err(|e| Error::Open(e.to_string()))?;
        Ok(PdfDocument { doc })
    }
}

/// An open PDF document, borrowing the [`PdfiumLibrary`] it was opened against.
pub struct PdfDocument<'a> {
    doc: pdfium_render::prelude::PdfDocument<'a>,
}

impl PdfDocument<'_> {
    /// Number of pages in the document.
    pub fn page_count(&self) -> usize {
        self.doc.pages().len() as usize
    }

    /// Size of the given page in PDF points.
    pub fn page_size(&self, index: usize) -> Result<PageSize> {
        let page = self.page(index)?;
        Ok(PageSize {
            width: page.width().value,
            height: page.height().value,
        })
    }

    /// Rasterizes a single page to an RGB image.
    pub fn render_page(&self, index: usize, opts: &RenderOptions) -> Result<PageImage> {
        let page = self.page(index)?;
        let scale = opts.scale();
        let px_w = (page.width().value * scale).round().max(1.0) as i32;
        let px_h = (page.height().value * scale).round().max(1.0) as i32;

        let config = PdfRenderConfig::new()
            .set_target_width(px_w)
            .set_target_height(px_h);

        let render = |e: pdfium_render::prelude::PdfiumError| Error::Render {
            page: index,
            message: e.to_string(),
        };
        let bitmap = page.render_with_config(&config).map_err(render)?;
        let image = bitmap.as_image().map_err(render)?.into_rgb8();
        Ok(PageImage {
            width: image.width(),
            height: image.height(),
            image,
        })
    }

    /// Rasterizes every page at the given options.
    pub fn render_all(&self, opts: &RenderOptions) -> Result<Vec<PageImage>> {
        (0..self.page_count())
            .map(|i| self.render_page(i, opts))
            .collect()
    }

    /// Extracts native (embedded) text from a page.
    ///
    /// Currently a stub returning no spans.
    // TODO(phase-native-text): reimplement pdftext-style word/line/span grouping.
    pub fn extract_text(&self, _index: usize) -> Result<Vec<PageText>> {
        Ok(Vec::new())
    }

    /// Fetches a page by index with a bounds check.
    fn page(&self, index: usize) -> Result<pdfium_render::prelude::PdfPage<'_>> {
        let count = self.page_count();
        if index >= count {
            return Err(Error::PageIndexOutOfRange { index, count });
        }
        self.doc.pages().get(index as i32).map_err(|e| Error::Render {
            page: index,
            message: e.to_string(),
        })
    }
}

/// Candidate PDFium library paths, most-specific first.
fn candidate_library_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(explicit) = std::env::var("MINERU_PDFIUM_LIB_PATH") {
        paths.push(PathBuf::from(explicit));
    }
    for dir in [
        "/Volumes/Archive/mineru/lib",
        "/opt/homebrew/lib",
        "/usr/local/lib",
    ] {
        paths.push(PathBuf::from(format!("{dir}/libpdfium.dylib")));
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    const DEMO_PDF: &str = "/Users/pohsuanlai/Documents/mineru/mineru/demo/pdfs/demo1.pdf";

    /// Serializes PDFium use across the parallel test harness (PDFium is not safe
    /// for concurrent access — see the crate-level Threading note).
    static PDFIUM_GUARD: Mutex<()> = Mutex::new(());

    // --- Pure unit tests (no native library required, always run) ---

    #[test]
    fn render_options_default_is_200_dpi() {
        assert_eq!(RenderOptions::default().dpi, 200.0);
        assert_eq!(DEFAULT_DPI, 200.0);
    }

    #[test]
    fn render_options_scale_is_dpi_over_72() {
        // 144 DPI is exactly twice the 72-DPI (1 px/pt) baseline.
        let opts = RenderOptions::with_dpi(144.0);
        assert!((opts.scale() - 2.0).abs() < 1e-6);
    }

    // --- Native-library tests (require a matching libpdfium at runtime) ---
    //
    // These are `#[ignore]`d by default. Binding to a missing or ABI-mismatched
    // Pdfium build aborts the process from inside the C library
    // (SIGSEGV/SIGTRAP) rather than returning an `Err`, so guarding on a failed
    // `PdfiumLibrary::load()` is not actually safe — the crash happens below the
    // Rust boundary. Run these only against a Pdfium build matching this crate's
    // selected `pdfium_*` feature:
    //
    //   MINERU_PDFIUM_LIB_PATH=/path/to/libpdfium.dylib \
    //     cargo test -p mineru-pdf -- --ignored

    #[test]
    #[ignore = "requires a matching libpdfium native library at runtime"]
    fn opens_counts_and_renders_demo1() {
        let _guard = PDFIUM_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let bytes = std::fs::read(DEMO_PDF).expect("demo pdf present");
        let lib = PdfiumLibrary::load().expect("load pdfium");
        let doc = lib.open(&bytes).expect("open demo pdf");

        assert!(doc.page_count() > 0, "demo1.pdf should have pages");

        let size = doc.page_size(0).expect("page 0 size");
        assert!(size.width > 0.0 && size.height > 0.0);

        let page = doc
            .render_page(0, &RenderOptions::default())
            .expect("render page 0");
        assert!(page.width > 0 && page.height > 0);
        // At 200 DPI the raster (px) should exceed the page (pts).
        assert!(page.width as f32 > size.width);

        let all = doc
            .render_all(&RenderOptions::default())
            .expect("render all pages");
        assert_eq!(all.len(), doc.page_count());
    }

    #[test]
    #[ignore = "requires a matching libpdfium native library at runtime"]
    fn out_of_range_page_errors() {
        let _guard = PDFIUM_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let bytes = std::fs::read(DEMO_PDF).expect("demo pdf present");
        let lib = PdfiumLibrary::load().expect("load pdfium");
        let doc = lib.open(&bytes).expect("open demo pdf");
        let n = doc.page_count();
        assert!(matches!(
            doc.page_size(n),
            Err(Error::PageIndexOutOfRange { .. })
        ));
    }
}
