//! PDF input for MinerU.
//!
//! Opens PDF bytes, reports page metadata, rasterizes pages to RGB images, and
//! extracts the native (embedded) text layer via the PDFium native library (bound
//! at runtime). Native-text extraction (the `pdftext`-style span/line grouping and
//! span back-fill) lives in [`text`]; see that module for the algorithm and the
//! PDF→top-left coordinate transform.
//!
//! # PDFium library resolution
//!
//! PDFium is loaded dynamically at runtime. [`PdfiumLibrary::load`] resolves the
//! native library in this order, logging (at INFO) which branch wins:
//!
//! 1. **`MINERU_PDFIUM_LIB_PATH` set:** if the file exists there, bind it. If it
//!    does not exist, auto-download a matching prebuilt PDFium to that exact path
//!    (creating parent dirs) and bind it.
//! 2. **`MINERU_PDFIUM_LIB_PATH` unset:** try common system locations
//!    (`/opt/homebrew/lib`, `/usr/local/lib`) and the platform default first; if
//!    none bind, auto-download to a per-user cache
//!    (`<MINERU_MODELS_DIR | $XDG_CACHE_HOME/mineru | $HOME/.cache/mineru>/pdfium/`)
//!    and bind from there, reusing the cached copy on later runs.
//!
//! No native library is bundled, keeping the crate small. The auto-download always
//! fetches the *latest* prebuilt build (see [`download`]) because the crate selects
//! pdfium-render's `pdfium_latest` feature and an ABI mismatch aborts the process.
//!
//! # Threading
//!
//! PDFium keeps process-global native state and is **not safe for concurrent use**
//! across threads, even through a single binding. [`PdfiumLibrary::load`] returns a
//! shared, bind-once instance, but callers must serialize actual document
//! operations — parse one document at a time, or guard access with a mutex. The
//! higher layers process documents sequentially for exactly this reason.

pub mod download;
pub mod error;
pub mod text;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use image::RgbImage;
use pdfium_render::prelude::{PdfRenderConfig, Pdfium};

pub use error::{Error, Result};
pub use text::{FilledRegion, Font, PageText, TextChar, TextLine, TextSpan};
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
    /// instance. The native library is resolved (and, if absent, auto-downloaded)
    /// as documented at the crate root. All of that — including any download —
    /// happens inside a single `get_or_init` under a lock, so the library is bound
    /// exactly once and never downloaded twice concurrently.
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

    /// Binds the native library, resolving (and if needed auto-downloading) it per
    /// the crate-root order.
    ///
    /// Runs entirely inside `load`'s single `get_or_init` (under a lock), so any
    /// download happens exactly once and the process binds exactly once.
    fn bind() -> Result<Self> {
        // 1. Explicit path via MINERU_PDFIUM_LIB_PATH.
        if let Some(explicit) = std::env::var_os("MINERU_PDFIUM_LIB_PATH") {
            let path = PathBuf::from(explicit);
            if path.exists() {
                tracing::info!(
                    path = %path.display(),
                    "using PDFium from MINERU_PDFIUM_LIB_PATH"
                );
            } else {
                tracing::info!(
                    path = %path.display(),
                    "MINERU_PDFIUM_LIB_PATH does not exist; auto-downloading PDFium there"
                );
                let url = download::download_pdfium_to(&path)?;
                tracing::info!(
                    url = %url,
                    path = %path.display(),
                    "downloaded PDFium to MINERU_PDFIUM_LIB_PATH"
                );
            }
            return Self::bind_path(&path);
        }

        // 2. No explicit path: try generic system locations, then the platform
        //    default, before falling back to a cached auto-download.
        for candidate in system_library_paths() {
            if let Ok(bindings) = Pdfium::bind_to_library(&candidate) {
                tracing::info!(path = %candidate.display(), "using PDFium from system location");
                return Ok(Self {
                    pdfium: Pdfium::new(bindings),
                });
            }
        }
        if let Ok(bindings) = Pdfium::bind_to_system_library() {
            tracing::info!("using PDFium from the platform default library search path");
            return Ok(Self {
                pdfium: Pdfium::new(bindings),
            });
        }

        // 3. Nothing resolved: auto-download into the per-user cache (reusing a
        //    prior download if present).
        let cached = cache_library_path()?;
        if cached.exists() {
            tracing::info!(path = %cached.display(), "using cached auto-downloaded PDFium");
        } else {
            tracing::info!(
                path = %cached.display(),
                "no PDFium found; auto-downloading to cache"
            );
            let url = download::download_pdfium_to(&cached)?;
            tracing::info!(url = %url, path = %cached.display(), "downloaded PDFium to cache");
        }
        Self::bind_path(&cached)
    }

    /// Binds PDFium from a concrete file path, mapping bind failure to
    /// [`Error::Bind`].
    fn bind_path(path: &Path) -> Result<Self> {
        Pdfium::bind_to_library(path)
            .map(|bindings| Self {
                pdfium: Pdfium::new(bindings),
            })
            .map_err(|e| Error::Bind(format!("binding {} failed: {e}", path.display())))
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

    /// Extracts the native (embedded) text layer of one page.
    ///
    /// Reads every glyph from the PDFium text page, deduplicates near-identical and
    /// shadow-offset duplicates, and groups the glyphs into spans and lines the way
    /// MinerU's `pdftext` path does. Returns an empty [`PageText`] for a page with
    /// no embedded text (a scanned page). See [`text`] for the algorithm and
    /// coordinate transform.
    pub fn extract_text(&self, index: usize) -> Result<PageText> {
        let page = self.page(index)?;
        text::extract_page_text(&page, index)
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

/// Generic cross-machine system locations to probe before auto-downloading.
///
/// Anything machine-specific must come through `MINERU_PDFIUM_LIB_PATH`, not a
/// baked-in path.
fn system_library_paths() -> Vec<PathBuf> {
    ["/opt/homebrew/lib", "/usr/local/lib"]
        .into_iter()
        .map(|dir| PathBuf::from(format!("{dir}/libpdfium.dylib")))
        .collect()
}

/// The per-user cache path an auto-downloaded PDFium is stored at.
///
/// Mirrors the cache-root convention used elsewhere in the workspace (see
/// `mineru-table`'s `weights.rs`): `MINERU_MODELS_DIR`, else
/// `$XDG_CACHE_HOME/mineru`, else `$HOME/.cache/mineru`. The library lives under
/// `<cache>/pdfium/<platform-specific filename>`.
fn cache_library_path() -> Result<PathBuf> {
    let root = if let Some(dir) = std::env::var_os("MINERU_MODELS_DIR") {
        PathBuf::from(dir)
    } else if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg).join("mineru")
    } else if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(home).join(".cache").join("mineru")
    } else {
        return Err(Error::Cache(
            "no writable cache directory: set MINERU_MODELS_DIR (or HOME)".to_string(),
        ));
    };
    let filename = download::current_asset()?.local_filename;
    Ok(root.join("pdfium").join(filename))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Directory holding the demo PDFs for the `#[ignore]`d native-lib tests.
    /// Set `MINERU_DEMO_DIR` to the repo's `demo/pdfs/` before running them.
    fn demo_dir() -> std::path::PathBuf {
        PathBuf::from(
            std::env::var("MINERU_DEMO_DIR")
                .expect("set MINERU_DEMO_DIR to the demo/pdfs directory"),
        )
    }

    /// Path to a demo PDF by file name, under [`demo_dir`].
    fn demo_pdf(name: &str) -> std::path::PathBuf {
        demo_dir().join(name)
    }

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
        let bytes = std::fs::read(demo_pdf("demo1.pdf")).expect("demo pdf present");
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
        let bytes = std::fs::read(demo_pdf("demo1.pdf")).expect("demo pdf present");
        let lib = PdfiumLibrary::load().expect("load pdfium");
        let doc = lib.open(&bytes).expect("open demo pdf");
        let n = doc.page_count();
        assert!(matches!(
            doc.page_size(n),
            Err(Error::PageIndexOutOfRange { .. })
        ));
    }

    /// Native-text extraction against a real digital (text-native) demo PDF.
    ///
    /// `demo1.pdf` has an embedded text layer; page 0 must extract non-empty text
    /// whose lines, read in order, contain the document's title/opening words.
    #[test]
    #[ignore = "requires a matching libpdfium native library at runtime"]
    fn extracts_native_text_from_digital_demo() {
        let _guard = PDFIUM_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let bytes = std::fs::read(demo_pdf("demo1.pdf")).expect("demo pdf present");
        let lib = PdfiumLibrary::load().expect("load pdfium");
        let doc = lib.open(&bytes).expect("open demo pdf");

        let page = doc.extract_text(0).expect("extract page 0 text");
        assert!(!page.chars.is_empty(), "digital PDF page 0 should have chars");
        assert!(page.supports_native_fill(), "upright digital page");
        assert!(!page.lines.is_empty(), "chars should group into lines");

        // Reading-order text of the page.
        let full: String = page
            .lines
            .iter()
            .map(|l| l.text())
            .collect::<Vec<_>>()
            .join("\n");
        // Print the first lines so a human can eyeball the real output.
        for line in page.lines.iter().take(8) {
            println!("LINE: {:?}", line.text());
        }
        // demo1.pdf is the paper "The response of flow duration curves to
        // afforestation"; assert its title words survive extraction, and that
        // "flow" precedes "afforestation" (reading order is preserved).
        let lower = full.to_lowercase();
        assert!(
            lower.contains("flow duration curves"),
            "expected title words; got: {}",
            &full.chars().take(400).collect::<String>()
        );
        assert!(lower.contains("afforestation"), "expected 'afforestation'");
        let flow_pos = lower.find("flow").expect("flow present");
        let affor_pos = lower.find("afforestation").expect("afforestation present");
        assert!(flow_pos < affor_pos, "reading order: 'flow' before 'afforestation'");
    }

    /// Probe helper: prints per-PDF char counts so we know which demo is digital.
    #[test]
    #[ignore = "diagnostic; requires libpdfium"]
    fn probe_which_demos_are_digital() {
        let _guard = PDFIUM_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let lib = PdfiumLibrary::load().expect("load pdfium");
        for name in ["demo1.pdf", "demo2.pdf", "demo3.pdf", "small_ocr.pdf"] {
            let bytes = std::fs::read(demo_pdf(name)).expect("demo pdf present");
            let doc = lib.open(&bytes).expect("open demo pdf");
            let page = doc.extract_text(0).expect("extract text");
            println!(
                "{name}: pages={} page0_chars={} page0_lines={} fillable={}",
                doc.page_count(),
                page.chars.len(),
                page.lines.len(),
                page.supports_native_fill(),
            );
        }
    }
}
