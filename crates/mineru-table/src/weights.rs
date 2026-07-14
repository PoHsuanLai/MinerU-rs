//! Runtime fetch + on-disk cache for the two table-recognition model weight
//! files (`.bpk`).
//!
//! The LCNet classifier and UNet segmenter are compiled into the crate (see
//! [`crate::generated`]), but their weights are downloaded once from a public
//! GitHub release and cached, rather than embedded. [`weight_path`] returns the
//! local path to a model's `.bpk`, fetching it on first use.
//!
//! ## Configuration (environment)
//!
//! - `MINERU_TABLE_WEIGHTS_BASE` — overrides the release base URL the `.bpk`
//!   files are fetched from (default [`DEFAULT_WEIGHTS_BASE`]). A trailing `/` is
//!   optional. The repository is public, so a plain unauthenticated HTTPS `GET`
//!   works; no token is required.
//! - `MINERU_MODELS_DIR` — root of the on-disk model cache. When unset, a
//!   per-user cache directory is used (`$XDG_CACHE_HOME/mineru` or
//!   `$HOME/.cache/mineru`). The `.bpk` files live under
//!   `<cache>/table-weights-v1/<filename>`.
//!
//! Everything here is panic-free and returns [`crate::error::Result`].

use std::io::Read;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Default base URL for the release the `.bpk` weight files are fetched from.
///
/// Overridable via the `MINERU_TABLE_WEIGHTS_BASE` environment variable.
pub const DEFAULT_WEIGHTS_BASE: &str =
    "https://github.com/PoHsuanLai/MinerU-rs/releases/download/table-weights-v1/";

/// Cache subdirectory (under the resolved cache root) holding this release's
/// weight files. Versioned so a future weight refresh lands in a fresh dir.
const CACHE_SUBDIR: &str = "table-weights-v1";

/// A table model whose weights are fetched at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableWeight {
    /// The PP-LCNet_x1_0 wired/wireless classifier.
    Lcnet,
    /// The UNet ruling-line segmenter.
    Unet,
}

impl TableWeight {
    /// The release asset / cached file name for this model's weights.
    fn filename(self) -> &'static str {
        match self {
            TableWeight::Lcnet => "lcnet_table_cls.bpk",
            TableWeight::Unet => "unet.bpk",
        }
    }

    /// The published SHA-256 digest (lowercase hex) of this model's `.bpk`, used
    /// to verify a freshly downloaded file.
    fn sha256(self) -> &'static str {
        match self {
            TableWeight::Lcnet => {
                "acebc6032282cee0f52dab1cf6ebc64c7fb7a6cdf8dd1317a8910db9d6db0277"
            }
            TableWeight::Unet => {
                "5483b824d8c8c243c368054bd09c3c49c006491dfd6a613c337dc64f9f89d08c"
            }
        }
    }
}

/// Resolves the local path to `model`'s `.bpk`, downloading and caching it on
/// first use.
///
/// If the file already exists in the cache it is returned as-is (no network
/// access). On a cache miss the file is fetched over HTTPS from the release base
/// URL, its SHA-256 is verified against the published digest, and it is written
/// atomically into the cache before the path is returned.
///
/// # Errors
///
/// - [`Error::Cache`] if no writable cache directory can be resolved, or a
///   filesystem operation fails.
/// - [`Error::WeightFetch`] if the download fails, returns a non-success status,
///   or the fetched bytes do not match the published SHA-256.
pub fn weight_path(model: TableWeight) -> Result<PathBuf> {
    let dir = cache_dir()?;
    let path = dir.join(model.filename());
    if path.exists() {
        return Ok(path);
    }

    // Ensure the cache subdirectory exists before writing into it.
    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::Cache(format!("failed to create cache dir {}: {e}", dir.display())))?;

    let bytes = fetch(model)?;
    verify_sha256(model, &bytes)?;

    // Write to a temp file in the same dir, then rename, so a concurrent reader
    // never sees a half-written `.bpk` (rename is atomic within a filesystem).
    let tmp = dir.join(format!("{}.{}.partial", model.filename(), std::process::id()));
    std::fs::write(&tmp, &bytes)
        .map_err(|e| Error::Cache(format!("failed to write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &path).map_err(|e| {
        // Best-effort cleanup of the temp file on failure.
        let _ = std::fs::remove_file(&tmp);
        Error::Cache(format!("failed to finalize {}: {e}", path.display()))
    })?;

    Ok(path)
}

/// Resolves the cache directory holding the weight files.
///
/// Order: `MINERU_MODELS_DIR`, else `XDG_CACHE_HOME/mineru`, else
/// `HOME/.cache/mineru`. The [`CACHE_SUBDIR`] release folder is appended.
fn cache_dir() -> Result<PathBuf> {
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
    Ok(root.join(CACHE_SUBDIR))
}

/// The base URL (with a guaranteed trailing `/`) weight files are fetched from.
fn weights_base() -> String {
    let base = std::env::var("MINERU_TABLE_WEIGHTS_BASE")
        .unwrap_or_else(|_| DEFAULT_WEIGHTS_BASE.to_string());
    if base.ends_with('/') {
        base
    } else {
        format!("{base}/")
    }
}

/// Downloads `model`'s `.bpk` over HTTPS and returns the raw bytes.
///
/// Uses `ureq` (synchronous, pure-Rust), which follows the release's 302 redirect
/// to the asset host automatically.
fn fetch(model: TableWeight) -> Result<Vec<u8>> {
    let url = format!("{}{}", weights_base(), model.filename());

    let resp = ureq::get(&url)
        .call()
        .map_err(|e| Error::WeightFetch(format!("GET {url} failed: {e}")))?;

    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| Error::WeightFetch(format!("reading body of {url} failed: {e}")))?;

    if bytes.is_empty() {
        return Err(Error::WeightFetch(format!("{url} returned an empty body")));
    }
    Ok(bytes)
}

/// Verifies `bytes` against `model`'s published SHA-256, erroring on mismatch.
fn verify_sha256(model: TableWeight, bytes: &[u8]) -> Result<()> {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let got = hex_lower(&digest);
    let want = model.sha256();
    if got != want {
        return Err(Error::WeightFetch(format!(
            "SHA-256 mismatch for {}: expected {want}, got {got}",
            model.filename()
        )));
    }
    Ok(())
}

/// Formats bytes as a lowercase hex string (no `hex` crate dependency).
fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        // `write!` to a String is infallible; format the two nibbles directly.
        const HEX: &[u8; 16] = b"0123456789abcdef";
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filenames_are_stable() {
        assert_eq!(TableWeight::Lcnet.filename(), "lcnet_table_cls.bpk");
        assert_eq!(TableWeight::Unet.filename(), "unet.bpk");
    }

    #[test]
    fn hex_lower_encodes_correctly() {
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }

    #[test]
    fn weights_base_has_trailing_slash() {
        // Default already ends with `/`; the helper preserves it.
        assert!(weights_base().ends_with('/'));
    }

    #[test]
    fn sha256_verify_rejects_wrong_bytes() {
        let err = verify_sha256(TableWeight::Lcnet, b"not the real weights");
        assert!(matches!(err, Err(Error::WeightFetch(_))));
    }
}
