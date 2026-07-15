//! Encoding and persistence of extracted image crops.
//!
//! Backends crop image/table/chart regions from the live page raster (using their
//! own croppers, which already know each backend's coordinate system) and hand the
//! resulting [`RgbImage`] here to be PNG-encoded and written through an
//! [`ImageWriter`](mineru_types::ImageWriter). Centralizing the encode + disk write
//! means only this crate needs the `image` crate's `png` feature; the backends
//! stay encode-free and, in their unit tests, disk-free (via a recording sink).

use std::path::PathBuf;

use image::{ImageFormat, RgbImage};
use mineru_types::ImageWriter;

/// Encodes an `RgbImage` to PNG bytes.
///
/// # Errors
/// Returns an [`std::io::Error`] if PNG encoding fails (e.g. an allocation or a
/// zero-dimension image); the `image` error is wrapped as [`std::io::ErrorKind::Other`].
pub fn encode_png(image: &RgbImage) -> std::io::Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut buf, ImageFormat::Png)
        .map_err(std::io::Error::other)?;
    Ok(buf.into_inner())
}

/// Encodes `image` to PNG and writes it to `sink` under `name` (a bare filename).
///
/// Convenience over [`encode_png`] + [`ImageWriter::write`] for the common case
/// where a backend already holds a cropped region.
///
/// # Errors
/// Propagates encoding errors from [`encode_png`] and write errors from the sink.
pub fn write_png(sink: &dyn ImageWriter, name: &str, image: &RgbImage) -> std::io::Result<()> {
    let bytes = encode_png(image)?;
    sink.write(name, &bytes)
}

/// An [`ImageWriter`] that persists crops as files under a fixed directory.
///
/// The directory is created lazily on first successful write's parent (callers
/// typically create it up front). Each `write(name, bytes)` writes `dir/name`.
#[derive(Debug, Clone)]
pub struct LocalFsImageWriter {
    dir: PathBuf,
}

impl LocalFsImageWriter {
    /// Creates a writer rooted at `dir`. The directory is not created here; the
    /// caller (or the first write's `create_dir_all`) ensures it exists.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

impl ImageWriter for LocalFsImageWriter {
    fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        // `name` is a bare filename by contract; join under the sink's directory.
        // Create the directory defensively so a write never fails on a missing dir.
        if let Some(parent) = self.dir.join(name).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(self.dir.join(name), bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// A recording sink for tests: captures `(name, byte_len)` without touching disk.
    #[derive(Debug, Default)]
    struct RecordingSink {
        writes: Mutex<Vec<(String, usize)>>,
    }
    impl ImageWriter for RecordingSink {
        fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
            self.writes
                .lock()
                .map_err(|_| std::io::Error::other("poisoned"))?
                .push((name.to_owned(), bytes.len()));
            Ok(())
        }
    }

    #[test]
    fn encode_png_produces_png_magic() {
        let img = RgbImage::from_pixel(4, 3, image::Rgb([10, 20, 30]));
        let bytes = encode_png(&img).expect("encode");
        // PNG magic number.
        assert_eq!(&bytes[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    #[test]
    fn write_png_routes_through_sink() {
        let sink = RecordingSink::default();
        let img = RgbImage::from_pixel(2, 2, image::Rgb([0, 0, 0]));
        write_png(&sink, "p1_o0.png", &img).expect("write");
        let writes = sink.writes.lock().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "p1_o0.png");
        assert!(writes[0].1 > 8, "wrote a non-trivial PNG");
    }

    #[test]
    fn local_fs_writer_writes_file() {
        let dir = std::env::temp_dir().join(format!("mineru-imgtest-{}", std::process::id()));
        let w = LocalFsImageWriter::new(&dir);
        w.write("a.png", b"hello").expect("write");
        let got = std::fs::read(dir.join("a.png")).expect("read back");
        assert_eq!(got, b"hello");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
