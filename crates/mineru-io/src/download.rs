//! Model-weight download helper, backed by [`hf-hub`](hf_hub).
//!
//! [`download_model`] performs a snapshot-style fetch: it lists the repository's
//! files and downloads each one into `cache_dir`, returning the local snapshot
//! directory. hf-hub is used in its synchronous (ureq) mode, so no async runtime
//! is required.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

use crate::error::{Error, Result};

/// Where model weights are fetched from.
///
/// Only [`Hugging Face`](ModelSource::HuggingFace) is implemented today; a
/// ModelScope mirror is planned as a fallback (see the `// TODO` in
/// [`download_model`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelSource {
    /// The Hugging Face Hub (`huggingface.co`).
    #[default]
    HuggingFace,
    /// The ModelScope mirror (`modelscope.cn`). Not yet implemented.
    ModelScope,
}

/// Snapshot-download every file in `repo_id` into `cache_dir`, returning the
/// local directory that holds the fetched snapshot.
///
/// hf-hub's cache is pointed at `cache_dir` (e.g. a large-storage models dir)
/// via [`ApiBuilder::with_cache_dir`], so downloads and reuse both live there.
/// The returned path is the directory containing the snapshot's files.
///
/// # Errors
///
/// Returns [`Error::Download`] if the Hub API cannot be built, the repository
/// listing fails, or any file fails to download. Selecting
/// [`ModelSource::ModelScope`] also returns [`Error::Download`] for now, as that
/// backend is not yet implemented.
///
/// # Network
///
/// This function performs network I/O; it is not exercised by the unit tests.
pub fn download_model(
    repo_id: &str,
    source: ModelSource,
    cache_dir: &Path,
) -> Result<PathBuf> {
    match source {
        ModelSource::HuggingFace => download_from_hf(repo_id, cache_dir),
        // TODO(phase-2): add a ModelScope fallback for regions where the HF Hub
        // is slow or unreachable. hf-hub only speaks to huggingface.co.
        ModelSource::ModelScope => Err(Error::Download(
            "ModelScope source is not yet implemented".to_string(),
        )),
    }
}

/// Fetch every file of `repo_id` from the Hugging Face Hub into `cache_dir`.
fn download_from_hf(repo_id: &str, cache_dir: &Path) -> Result<PathBuf> {
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| Error::Download(format!("failed to build HF Hub API: {e}")))?;

    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let info = repo
        .info()
        .map_err(|e| Error::Download(format!("failed to list repo {repo_id}: {e}")))?;

    let mut snapshot_dir: Option<PathBuf> = None;
    for sibling in &info.siblings {
        let path = repo.get(&sibling.rfilename).map_err(|e| {
            Error::Download(format!(
                "failed to download {}/{}: {e}",
                repo_id, sibling.rfilename
            ))
        })?;
        // Every file resolves under the same snapshot directory; remember the
        // parent of the first one as the directory to return.
        if snapshot_dir.is_none() {
            snapshot_dir = path.parent().map(Path::to_path_buf);
        }
    }

    // A repo with no files still counts as a successful (empty) snapshot; fall
    // back to the cache root in that case.
    Ok(snapshot_dir.unwrap_or_else(|| cache_dir.to_path_buf()))
}
