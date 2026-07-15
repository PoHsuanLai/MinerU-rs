//! Filesystem I/O and model-weight download helpers for MinerU.
//!
//! This crate provides a small, object-safe reader/writer abstraction over the
//! local filesystem ([`DataReader`] / [`DataWriter`], implemented by
//! [`LocalFsReader`] / [`LocalFsWriter`]), an output-directory layout helper
//! ([`prepare_output_dirs`]) mirroring Python's `prepare_env`, and a model
//! snapshot-download helper ([`download_model`]) backed by `hf-hub`.
//!
//! There is deliberately **no** object-storage (S3) backend: MinerU-rs reads and
//! writes the local disk only.

pub mod data;
pub mod download;
pub mod error;
pub mod image_crop;
pub mod layout;

pub use data::{DataReader, DataWriter, LocalFsReader, LocalFsWriter};
pub use download::{download_model, ModelSource};
pub use error::{Error, Result};
pub use image_crop::{encode_png, write_png, LocalFsImageWriter};
pub use layout::{prepare_output_dirs, OutputLayout};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_fs_round_trip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let writer = LocalFsWriter::new(dir.path());
        let reader = LocalFsReader::new(dir.path());

        // Nested path exercises auto-creation of parent directories.
        writer.write("sub/dir/hello.bin", b"binary payload")?;
        writer.write_string("sub/dir/hello.txt", "text payload")?;

        assert_eq!(reader.read("sub/dir/hello.bin")?, b"binary payload");
        assert_eq!(reader.read("sub/dir/hello.txt")?, b"text payload");
        Ok(())
    }

    #[test]
    fn prepare_output_dirs_creates_images_subdir() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let layout = prepare_output_dirs(dir.path(), "mydoc", "auto")?;

        assert_eq!(layout.base, dir.path().join("mydoc").join("auto"));
        assert_eq!(layout.images, layout.base.join("images"));
        assert!(layout.base.is_dir());
        assert!(layout.images.is_dir());
        Ok(())
    }

    #[test]
    fn write_rejects_parent_dir_escape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let writer = LocalFsWriter::new(dir.path());

        let err = writer.write("../escape.txt", b"nope").unwrap_err();
        assert!(matches!(err, Error::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn read_rejects_absolute_path_escape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let reader = LocalFsReader::new(dir.path());

        let err = reader.read("/etc/passwd").unwrap_err();
        assert!(matches!(err, Error::PathEscape(_)), "got {err:?}");
    }

    #[test]
    #[ignore = "hits the network and writes to disk; run explicitly"]
    fn download_model_smoke() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = download_model("hf-internal-testing/tiny-random-bert", ModelSource::HuggingFace, dir.path())?;
        assert!(path.exists());
        Ok(())
    }
}
