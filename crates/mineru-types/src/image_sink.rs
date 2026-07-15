//! The [`ImageWriter`] sink: where a backend hands off extracted image crops.
//!
//! Cropping an image/table/chart region requires the page raster, which lives only
//! inside a backend's per-page loop and never crosses the [`Backend`](crate::Backend)
//! seam. So the *caller* (the CLI's run flow) supplies a sink — rooted at the output
//! `images/` directory — via [`ParseOptions::image_sink`](crate::ParseOptions), and
//! the backend calls [`ImageWriter::write`] with the already-encoded bytes.
//!
//! The sink takes encoded bytes (not an `RgbImage`) so this base crate stays free of
//! an `image` dependency; encoding lives alongside the concrete disk writer in
//! `mineru-io`. Tests use a recording fake instead of touching disk.

/// A sink for extracted image bytes, keyed by a bare file name.
///
/// The name is a bare filename (no directory) such as `p2_o0.png`; the sink owns
/// the target directory. Implementors persist (or, in tests, record) the bytes.
pub trait ImageWriter: std::fmt::Debug + Send + Sync {
    /// Persists `bytes` under `name` (a bare filename, no directory component).
    ///
    /// # Errors
    /// Returns any I/O error from persisting. Callers treat a write failure as
    /// non-fatal to the parse (a missing crop degrades output, it does not abort).
    fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()>;
}
