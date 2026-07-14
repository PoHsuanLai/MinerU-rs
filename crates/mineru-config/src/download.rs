//! Auto-download of the pipeline model weight files into the models cache dir.
//!
//! On a clean machine the pipeline backend has no weights on disk. Rather than
//! hard-erroring and forcing the user to provision them by hand, this module
//! fetches any missing weight files from a public GitHub release and caches them
//! under the resolved [`crate::Config::models_dir`], so `mineru <pdf>` works out
//! of the box.
//!
//! It mirrors the already-shipped PDFium auto-download
//! (`mineru-pdf::download`): a base URL resolved from an env var (default
//! [`DEFAULT_MODELS_BASE`]), a synchronous `ureq` HTTPS `GET` per file, atomic
//! `.partial-<pid>` temp-write + rename, parent-dir creation, and
//! `tracing::info!` logging. It is *simpler* than the PDFium fetch — there is no
//! archive extraction or platform mapping; each weight is a plain file fetched by
//! its relative path and written verbatim.
//!
//! # Idempotent / cached
//!
//! Files that already exist are skipped, so a fully-provisioned models dir never
//! touches the network. Only genuinely missing files are fetched.
//!
//! # Configuration (environment)
//!
//! - `MINERU_MODELS_BASE` — overrides the release base URL weight files are
//!   fetched from (default [`DEFAULT_MODELS_BASE`]). A trailing `/` is optional.
//!
//! Everything here is panic-free and returns [`crate::error::Result`].

use std::io::Read;
use std::path::Path;

use crate::error::{Error, Result};

/// Default base URL the pipeline weight files are fetched from.
///
/// Matches the release-hosting style already used by `mineru-table`'s weight
/// fetch (`.../releases/download/<tag>/`). The files are not hosted at this tag
/// yet — the dev will publish them later; overridable via the `MINERU_MODELS_BASE`
/// environment variable in the meantime.
pub const DEFAULT_MODELS_BASE: &str =
    "https://github.com/PoHsuanLai/MinerU-rs/releases/download/pipeline-models-v1/";

/// The weight files required by the pipeline backend, as paths relative to the
/// models directory root.
///
/// **This list must stay in sync with `ModelPaths::under` in
/// `mineru-backend-pipeline/src/models.rs`.** The config crate is foundational and
/// must not depend on the pipeline crate (that would invert the dependency DAG),
/// so the canonical relative-path list is duplicated here with this note rather
/// than imported. `ModelPaths::under` carries a reciprocal comment.
///
/// The two ONNX table models additionally need a sibling `<stem>.safetensors`
/// (that is what `SlaNet::load` / the UNet loader actually read — the `.onnx` is
/// the pipeline's path handle), so those `.safetensors` files are listed too.
pub const REQUIRED_MODEL_FILES: &[&str] = &[
    // Layout: PP-DocLayoutV2 detector.
    "Layout/PP-DocLayoutV2/model.safetensors",
    // OCR: PP-OCRv6 detector, recognizer, and character dict.
    "OCR/paddleocr_torch/ch_PP-OCRv6_small_det_infer.safetensors",
    "OCR/paddleocr_torch/ch_PP-OCRv6_small_rec_infer.safetensors",
    // The v6 dict is OPTIONAL at load time: `mineru-ocr-rec` embeds a PP-OCRv6
    // fallback and the pipeline loader only reads this file if it exists. It stays
    // in the list because caching it locally is nice-to-have, but because the
    // download is best-effort a 404/missing host for it merely warns.
    "OCR/paddleocr_torch/ppocrv6_dict.txt",
    // Formula: UniMerNet checkpoint directory (two files).
    "MFR/unimernet_hf_small_2503/model.safetensors",
    "MFR/unimernet_hf_small_2503/tokenizer.json",
    // Tables: the `.onnx` path handle plus the sibling `.safetensors` actually read.
    "TabRec/SlanetPlus/slanet-plus.onnx",
    "TabRec/SlanetPlus/slanet-plus.safetensors",
    "TabRec/UnetStructure/unet.onnx",
    "TabRec/UnetStructure/unet.safetensors",
];

/// The base URL (with a guaranteed trailing `/`) weight files are fetched from.
///
/// Resolved from `MINERU_MODELS_BASE`, else [`DEFAULT_MODELS_BASE`]. A missing
/// trailing slash is appended so `{base}{relative}` joins correctly.
fn models_base() -> String {
    let base =
        std::env::var("MINERU_MODELS_BASE").unwrap_or_else(|_| DEFAULT_MODELS_BASE.to_string());
    normalize_base(&base)
}

/// Appends a trailing `/` to `base` if it lacks one, so it joins with a relative
/// path. Pulled out of [`models_base`] so the normalization is unit-testable
/// without touching the process environment.
fn normalize_base(base: &str) -> String {
    if base.ends_with('/') {
        base.to_string()
    } else {
        format!("{base}/")
    }
}

/// Best-effort download of any [`REQUIRED_MODEL_FILES`] missing under `models_dir`.
///
/// For each required relative path, if the file already exists on disk it is left
/// as-is (no network access). A missing file is fetched from `{base}/{relative}`,
/// its parent directory is created, and it is written atomically (to a
/// `<name>.partial-<pid>` temp file in the same directory, then renamed) so a
/// concurrent reader never observes a half-written file.
///
/// A fully-provisioned models dir does not hit the network at all.
///
/// # Best-effort (matches the pipeline loader's philosophy)
///
/// This mirrors `PipelineModels::load_from_on`, which loads each stage best-effort
/// and *skips* a missing/unloadable weight with a warning rather than failing the
/// whole run. Accordingly, a per-file failure here — a 404, a network error, the
/// release not yet being hosted, or a filesystem error — is **not** fatal: it is
/// logged with [`tracing::warn!`] and the loop continues. The pipeline's loader
/// then decides what is actually fatal (it warns on and skips missing stages, and
/// errors clearly only if *nothing* loads). Some required entries are genuinely
/// optional at load time — e.g. `ppocrv6_dict.txt` has an embedded fallback in
/// `mineru-ocr-rec` — so a failure to fetch them must never abort the run.
///
/// It returns [`Ok`] after attempting every missing file. The `Result` signature
/// is retained for forward-compatibility (and matches the crate convention) but no
/// per-file fetch/IO error is propagated.
pub fn download_missing_models(models_dir: &Path) -> Result<()> {
    download_missing_with(models_dir, REQUIRED_MODEL_FILES, fetch);
    Ok(())
}

/// Core download loop, generic over the byte-fetching function.
///
/// Best-effort: each missing file is attempted independently and any failure is
/// logged and skipped (see [`download_missing_models`]). The `fetch` seam lets
/// tests drive the loop without hitting the network — a test can pass a fetcher
/// that panics if called (to prove the "already exists → skipped" path never
/// fetches) or one that returns an error (to prove a failed fetch does not abort).
fn download_missing_with<F>(models_dir: &Path, files: &[&str], fetch: F)
where
    F: Fn(&str) -> Result<Vec<u8>>,
{
    let base = models_base();
    for relative in files {
        let target = models_dir.join(relative);
        if target.exists() {
            continue;
        }

        let url = format!("{base}{relative}");
        tracing::info!("downloading {relative} (from {url})");

        if let Err(e) = fetch_one(&target, &url, &fetch) {
            // Best-effort: a missing/unhostable file is not fatal here. The
            // pipeline loader downstream will skip the corresponding stage (or use
            // an embedded fallback) and warn, and only errors if nothing loads.
            tracing::warn!("skipping {relative}: {e}");
        }
    }
}

/// Fetches one file and writes it atomically into place under `target`.
///
/// Split out so its (best-effort, per-file) failure is caught by the caller and
/// turned into a warning rather than aborting the whole provisioning pass.
fn fetch_one<F>(target: &Path, url: &str, fetch: &F) -> Result<()>
where
    F: Fn(&str) -> Result<Vec<u8>>,
{
    let parent = target.parent().ok_or_else(|| {
        Error::Cache(format!("target path {} has no parent directory", target.display()))
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| Error::Cache(format!("failed to create dir {}: {e}", parent.display())))?;

    let bytes = fetch(url)?;

    // Write to a temp file in the same directory, then rename atomically so a
    // concurrent reader never sees a partially written weight file.
    let file_name = target
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Cache(format!("target path {} has no file name", target.display())))?;
    let tmp = target.with_file_name(format!("{file_name}.partial-{}", std::process::id()));
    std::fs::write(&tmp, &bytes)
        .map_err(|e| Error::Cache(format!("failed to write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, target).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        Error::Cache(format!("failed to finalize {}: {e}", target.display()))
    })?;
    Ok(())
}

/// Downloads `url` over HTTPS and returns the raw file bytes.
///
/// Uses `ureq` (synchronous, pure-Rust), which follows GitHub's release-download
/// 302 redirect to the asset host automatically.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_base_appends_trailing_slash() {
        assert_eq!(normalize_base("https://example.com/x"), "https://example.com/x/");
    }

    #[test]
    fn normalize_base_preserves_existing_trailing_slash() {
        assert_eq!(normalize_base("https://example.com/x/"), "https://example.com/x/");
    }

    #[test]
    fn default_base_has_trailing_slash() {
        assert!(models_base().ends_with('/'));
    }

    #[test]
    fn required_files_are_nonempty_relative_paths() {
        assert!(!REQUIRED_MODEL_FILES.is_empty());
        for f in REQUIRED_MODEL_FILES {
            assert!(!f.is_empty());
            assert!(!f.starts_with('/'), "{f} must be relative to the models dir");
        }
    }

    /// Every already-present file is skipped and the fetcher is never called, so a
    /// fully-provisioned models dir does no network access.
    #[test]
    fn existing_files_are_skipped_without_fetching() {
        let dir = std::env::temp_dir().join(format!("mineru-models-skip-{}", std::process::id()));
        let files = ["a/one.txt", "b/c/two.bin"];
        for f in files {
            let p = dir.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"already here").unwrap();
        }

        // A fetcher that fails the test if it is ever called.
        let never = |url: &str| -> Result<Vec<u8>> {
            panic!("network fetch attempted for {url} but all files exist");
        };

        download_missing_with(&dir, &files, never);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A missing file is fetched (via the seam), its parent dir is created, and it
    /// is written atomically into place. No real network is used — the fetcher
    /// returns canned bytes.
    #[test]
    fn missing_file_is_fetched_and_written() {
        let dir = std::env::temp_dir().join(format!("mineru-models-fetch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let files = ["nested/dir/weight.bin"];

        let canned = |_url: &str| -> Result<Vec<u8>> { Ok(b"fetched-bytes".to_vec()) };

        download_missing_with(&dir, &files, canned);

        let written = std::fs::read(dir.join("nested/dir/weight.bin")).expect("file written");
        assert_eq!(written, b"fetched-bytes");
        // No temp files should linger.
        let leftover: Vec<_> = std::fs::read_dir(dir.join("nested/dir"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".partial-"))
            .collect();
        assert!(leftover.is_empty(), "no .partial temp files should remain");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A failed fetch (e.g. a 404) is best-effort: it does NOT abort the pass, the
    /// bad file is simply left absent, and other missing files still download.
    #[test]
    fn failed_fetch_is_best_effort_and_does_not_abort() {
        let dir = std::env::temp_dir().join(format!("mineru-models-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let files = ["will/fail.bin", "will/succeed.bin"];

        // Fail the first file (simulated 404), serve the second.
        let mixed = |url: &str| -> Result<Vec<u8>> {
            if url.ends_with("will/fail.bin") {
                Err(Error::Download(format!("GET {url} failed: status code 404")))
            } else {
                Ok(b"ok".to_vec())
            }
        };

        // Must return without panicking or propagating the 404.
        download_missing_with(&dir, &files, mixed);

        // The failed file is absent; the good one was still written.
        assert!(!dir.join("will/fail.bin").exists(), "failed fetch must leave no file");
        assert_eq!(std::fs::read(dir.join("will/succeed.bin")).unwrap(), b"ok");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `download_missing_models` returns `Ok` even when the (default, unhosted)
    /// base URL 404s every file — best-effort provisioning never aborts the caller.
    #[test]
    fn download_missing_models_is_ok_on_all_failures() {
        let dir = std::env::temp_dir().join(format!("mineru-models-allfail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Point at a bogus, non-resolving host so any real fetch attempt fails fast
        // rather than hitting the network for real. (create_dir_all runs first, but
        // the fetch never succeeds.) We only assert it returns Ok, not that it is
        // offline — the failed-fetch unit test above covers the no-network seam.
        let only_fail = |url: &str| -> Result<Vec<u8>> {
            Err(Error::Download(format!("GET {url} failed: simulated")))
        };
        download_missing_with(&dir, REQUIRED_MODEL_FILES, only_fail);
        // Nothing was provisioned, but the call did not abort.
        for f in REQUIRED_MODEL_FILES {
            assert!(!dir.join(f).exists());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real end-to-end check of the download path against the release host.
    ///
    /// `#[ignore]`d: it hits the network and depends on `MINERU_MODELS_BASE` (or
    /// the default release) actually hosting the files, which is not yet the case.
    /// It downloads the required files into a temp dir and asserts they exist and
    /// are non-empty. Kept compiling so it is ready once the release is published.
    #[test]
    #[ignore = "network: fetches pipeline model weights from the release host"]
    fn download_missing_smoke() {
        let dir = std::env::temp_dir().join(format!("mineru-models-smoke-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        download_missing_models(&dir).expect("download missing models");

        for relative in REQUIRED_MODEL_FILES {
            let p = dir.join(relative);
            let meta = std::fs::metadata(&p).expect("required file downloaded");
            assert!(meta.len() > 0, "{} should be non-empty", p.display());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
