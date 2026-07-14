//! Auto-download + extract of a prebuilt PDFium native library.
//!
//! When no PDFium library can be resolved from an explicit path or the system,
//! [`crate::PdfiumLibrary`] falls back to fetching a matching prebuilt binary and
//! caching it on disk. This module owns that fetch: it maps the running
//! platform/architecture to the correct [bblanchon/pdfium-binaries] release asset,
//! downloads it, and extracts the single native-library entry to a target path.
//!
//! # ABI safety
//!
//! Binding to an ABI-mismatched PDFium build aborts the process from *inside* the
//! C library (SIGSEGV/SIGTRAP) rather than returning an error. The crate selects
//! pdfium-render's `pdfium_latest` feature, so the download **must** fetch the
//! *latest* pdfium build. That is why the default base URL points at the
//! `releases/latest/download/` path — do not point it at a pinned older release.
//!
//! # Configuration (environment)
//!
//! - `MINERU_PDFIUM_DOWNLOAD_BASE` — overrides the release base URL assets are
//!   fetched from (default [`DEFAULT_DOWNLOAD_BASE`]). A trailing `/` is optional.
//!
//! [bblanchon/pdfium-binaries]: https://github.com/bblanchon/pdfium-binaries/releases
//!
//! Everything here is panic-free and returns [`crate::error::Result`].

use std::io::Read;
use std::path::Path;

use crate::error::{Error, Result};

/// Default base URL prebuilt PDFium archives are fetched from.
///
/// The `latest/download/` path 302-redirects to the newest versioned asset, which
/// is the build pdfium-render's `pdfium_latest` feature targets. Overridable via
/// the `MINERU_PDFIUM_DOWNLOAD_BASE` environment variable.
pub const DEFAULT_DOWNLOAD_BASE: &str =
    "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/";

/// The prebuilt-asset descriptor for one platform/architecture: which archive to
/// fetch, where the native library lives inside it, and what to name it on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfiumAsset {
    /// The bblanchon release asset file name (e.g. `pdfium-mac-arm64.tgz`).
    pub asset_name: &'static str,
    /// Path of the native library *inside* the archive (e.g. `lib/libpdfium.dylib`).
    pub archive_lib_path: &'static str,
    /// The local library file name for this platform (e.g. `libpdfium.dylib`).
    pub local_filename: &'static str,
    /// Whether the archive is a `.zip` (Windows) rather than a gzipped tar.
    pub is_zip: bool,
}

/// Resolves the prebuilt PDFium asset for the running platform/architecture.
///
/// Uses [`std::env::consts::OS`] and [`std::env::consts::ARCH`] at runtime.
/// Returns [`Error::UnsupportedPlatform`] for any OS/arch without a known
/// bblanchon asset — never guesses, because a wrong binary aborts the process on
/// bind (see the module ABI-safety note).
pub fn current_asset() -> Result<PdfiumAsset> {
    asset_for(std::env::consts::OS, std::env::consts::ARCH)
}

/// Maps an explicit `(os, arch)` pair to its prebuilt asset descriptor.
///
/// Split out from [`current_asset`] so the mapping is unit-testable without
/// depending on the host the tests happen to run on.
pub fn asset_for(os: &str, arch: &str) -> Result<PdfiumAsset> {
    let asset = match (os, arch) {
        ("macos", "aarch64") => PdfiumAsset {
            asset_name: "pdfium-mac-arm64.tgz",
            archive_lib_path: "lib/libpdfium.dylib",
            local_filename: "libpdfium.dylib",
            is_zip: false,
        },
        ("macos", "x86_64") => PdfiumAsset {
            asset_name: "pdfium-mac-x64.tgz",
            archive_lib_path: "lib/libpdfium.dylib",
            local_filename: "libpdfium.dylib",
            is_zip: false,
        },
        ("linux", "x86_64") => PdfiumAsset {
            asset_name: "pdfium-linux-x64.tgz",
            archive_lib_path: "lib/libpdfium.so",
            local_filename: "libpdfium.so",
            is_zip: false,
        },
        ("linux", "aarch64") => PdfiumAsset {
            asset_name: "pdfium-linux-arm64.tgz",
            archive_lib_path: "lib/libpdfium.so",
            local_filename: "libpdfium.so",
            is_zip: false,
        },
        ("windows", "x86_64") => PdfiumAsset {
            asset_name: "pdfium-win-x64.zip",
            archive_lib_path: "bin/pdfium.dll",
            local_filename: "pdfium.dll",
            is_zip: true,
        },
        _ => {
            return Err(Error::UnsupportedPlatform(format!(
                "no prebuilt PDFium asset for os={os} arch={arch}"
            )));
        }
    };
    Ok(asset)
}

/// The base URL (with a guaranteed trailing `/`) assets are fetched from.
fn download_base() -> String {
    let base = std::env::var("MINERU_PDFIUM_DOWNLOAD_BASE")
        .unwrap_or_else(|_| DEFAULT_DOWNLOAD_BASE.to_string());
    if base.ends_with('/') {
        base
    } else {
        format!("{base}/")
    }
}

/// Downloads the prebuilt PDFium archive for the running platform and extracts its
/// single native-library entry to `target`.
///
/// The target's parent directory is created if needed. The library is written
/// atomically: it is extracted to a `<target>.partial-<pid>` temp file in the same
/// directory and then renamed into place, so a concurrent reader never observes a
/// half-written library. Returns the fetched archive's URL (for logging).
///
/// Does **not** bind the resulting library (binding an ABI-mismatched build aborts
/// the process — see the module note). Callers bind via the normal
/// `Pdfium::bind_to_library` path after this returns.
///
/// # Errors
///
/// - [`Error::UnsupportedPlatform`] if the host has no known prebuilt asset.
/// - [`Error::Cache`] if the target directory / temp file cannot be created.
/// - [`Error::Download`] if the HTTP fetch fails or returns an empty body.
/// - [`Error::Unpack`] if the archive cannot be read or lacks the library entry.
pub fn download_pdfium_to(target: &Path) -> Result<String> {
    let asset = current_asset()?;
    let url = format!("{}{}", download_base(), asset.asset_name);

    let parent = target.parent().ok_or_else(|| {
        Error::Cache(format!(
            "target path {} has no parent directory",
            target.display()
        ))
    })?;
    std::fs::create_dir_all(parent).map_err(|e| {
        Error::Cache(format!("failed to create dir {}: {e}", parent.display()))
    })?;

    let bytes = fetch(&url)?;
    let lib_bytes = extract_library(&asset, &bytes)?;

    // Extract to a temp file in the same directory, then rename atomically so a
    // concurrent reader never sees a partially written library.
    let tmp = target.with_file_name(format!(
        "{}.partial-{}",
        asset.local_filename,
        std::process::id()
    ));
    std::fs::write(&tmp, &lib_bytes)
        .map_err(|e| Error::Cache(format!("failed to write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, target).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        Error::Cache(format!("failed to finalize {}: {e}", target.display()))
    })?;

    Ok(url)
}

/// Downloads `url` over HTTPS and returns the raw archive bytes.
///
/// Uses `ureq` (synchronous, pure-Rust), which follows GitHub's `latest/download`
/// 302 redirect to the versioned asset host automatically.
fn fetch(url: &str) -> Result<Vec<u8>> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| Error::Download(format!("GET {url} failed: {e}")))?;

    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Download(format!("reading body of {url} failed: {e}")))?;

    if bytes.is_empty() {
        return Err(Error::Download(format!("{url} returned an empty body")));
    }
    Ok(bytes)
}

/// Extracts the single native-library entry named `asset.archive_lib_path` from
/// the in-memory archive `bytes`, returning its raw contents.
fn extract_library(asset: &PdfiumAsset, bytes: &[u8]) -> Result<Vec<u8>> {
    if asset.is_zip {
        extract_from_zip(asset.archive_lib_path, bytes)
    } else {
        extract_from_tgz(asset.archive_lib_path, bytes)
    }
}

/// Extracts one entry from a gzipped tar (`.tgz`) archive held in memory.
fn extract_from_tgz(entry_path: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    let decoder = flate2::read::GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|e| Error::Unpack(format!("reading tar entries failed: {e}")))?;

    for entry in entries {
        let mut entry =
            entry.map_err(|e| Error::Unpack(format!("reading tar entry failed: {e}")))?;
        let path = entry
            .path()
            .map_err(|e| Error::Unpack(format!("reading tar entry path failed: {e}")))?;
        if path.to_string_lossy() == entry_path {
            let mut out = Vec::new();
            entry
                .read_to_end(&mut out)
                .map_err(|e| Error::Unpack(format!("reading {entry_path} failed: {e}")))?;
            if out.is_empty() {
                return Err(Error::Unpack(format!("{entry_path} is empty in archive")));
            }
            return Ok(out);
        }
    }
    Err(Error::Unpack(format!(
        "archive did not contain {entry_path}"
    )))
}

/// Extracts one entry from a `.zip` archive held in memory (Windows assets).
#[cfg(windows)]
fn extract_from_zip(entry_path: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    let reader = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| Error::Unpack(format!("opening zip failed: {e}")))?;
    let mut file = archive
        .by_name(entry_path)
        .map_err(|e| Error::Unpack(format!("archive did not contain {entry_path}: {e}")))?;
    let mut out = Vec::new();
    file.read_to_end(&mut out)
        .map_err(|e| Error::Unpack(format!("reading {entry_path} failed: {e}")))?;
    if out.is_empty() {
        return Err(Error::Unpack(format!("{entry_path} is empty in archive")));
    }
    Ok(out)
}

/// Non-Windows stub: `.zip` assets only exist for Windows, and the `zip` crate is
/// a Windows-only dependency, so this branch is unreachable elsewhere.
#[cfg(not(windows))]
fn extract_from_zip(_entry_path: &str, _bytes: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Unpack(
        "zip archives are only supported on Windows".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_arm64_asset_mapping() {
        let a = asset_for("macos", "aarch64").expect("mac arm64 supported");
        assert_eq!(a.asset_name, "pdfium-mac-arm64.tgz");
        assert_eq!(a.archive_lib_path, "lib/libpdfium.dylib");
        assert_eq!(a.local_filename, "libpdfium.dylib");
        assert!(!a.is_zip);
    }

    #[test]
    fn mac_x64_asset_mapping() {
        let a = asset_for("macos", "x86_64").expect("mac x64 supported");
        assert_eq!(a.asset_name, "pdfium-mac-x64.tgz");
        assert_eq!(a.archive_lib_path, "lib/libpdfium.dylib");
        assert_eq!(a.local_filename, "libpdfium.dylib");
        assert!(!a.is_zip);
    }

    #[test]
    fn linux_x64_asset_mapping() {
        let a = asset_for("linux", "x86_64").expect("linux x64 supported");
        assert_eq!(a.asset_name, "pdfium-linux-x64.tgz");
        assert_eq!(a.archive_lib_path, "lib/libpdfium.so");
        assert_eq!(a.local_filename, "libpdfium.so");
        assert!(!a.is_zip);
    }

    #[test]
    fn linux_arm64_asset_mapping() {
        let a = asset_for("linux", "aarch64").expect("linux arm64 supported");
        assert_eq!(a.asset_name, "pdfium-linux-arm64.tgz");
        assert_eq!(a.archive_lib_path, "lib/libpdfium.so");
    }

    #[test]
    fn windows_x64_asset_mapping() {
        let a = asset_for("windows", "x86_64").expect("windows x64 supported");
        assert_eq!(a.asset_name, "pdfium-win-x64.zip");
        assert_eq!(a.archive_lib_path, "bin/pdfium.dll");
        assert_eq!(a.local_filename, "pdfium.dll");
        assert!(a.is_zip);
    }

    #[test]
    fn unknown_arch_is_unsupported() {
        let err = asset_for("macos", "riscv64");
        assert!(matches!(err, Err(Error::UnsupportedPlatform(_))));
    }

    #[test]
    fn unknown_os_is_unsupported() {
        let err = asset_for("plan9", "x86_64");
        assert!(matches!(err, Err(Error::UnsupportedPlatform(_))));
    }

    #[test]
    fn download_base_has_trailing_slash() {
        assert!(download_base().ends_with('/'));
    }

    /// Real end-to-end check of the download + extract path against bblanchon.
    ///
    /// `#[ignore]`d: it hits the network, pulls ~10 MB, and depends on GitHub
    /// releases being reachable. It downloads and extracts into a temp dir and
    /// asserts the file exists, is non-trivial, and starts with the platform's
    /// native-library magic bytes. It deliberately does **not** bind the library
    /// (an ABI mismatch would abort the process — the whole-pipeline run exercises
    /// binding separately).
    #[test]
    #[ignore = "network: downloads ~10MB from bblanchon/pdfium-binaries"]
    fn download_and_extract_smoke() {
        let asset = current_asset().expect("host platform supported");
        let dir = std::env::temp_dir().join(format!("mineru-pdfium-smoke-{}", std::process::id()));
        let target = dir.join(asset.local_filename);

        let url = download_pdfium_to(&target).expect("download + extract");
        println!("downloaded from {url} -> {}", target.display());

        let bytes = std::fs::read(&target).expect("read extracted library");
        println!("extracted {} bytes", bytes.len());
        assert!(bytes.len() > 1_000_000, "library should be >1MB, got {}", bytes.len());

        // Magic-byte sanity: Mach-O (mac), ELF (linux), or PE/MZ (windows).
        let magic = &bytes[..4.min(bytes.len())];
        let ok = match std::env::consts::OS {
            // Mach-O: 0xFEEDFACF (64-bit) or reversed 0xCFFAEDFE little-endian.
            "macos" => magic == [0xCF, 0xFA, 0xED, 0xFE] || magic == [0xFE, 0xED, 0xFA, 0xCF],
            // ELF: 0x7F 'E' 'L' 'F'.
            "linux" => magic == [0x7F, b'E', b'L', b'F'],
            // PE: starts with 'M' 'Z'.
            "windows" => &magic[..2] == b"MZ",
            _ => false,
        };
        assert!(ok, "unexpected magic bytes {magic:02x?} for {}", std::env::consts::OS);

        // Best-effort cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }
}
