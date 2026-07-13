//! Output directory layout, mirroring Python's `prepare_env`.
//!
//! MinerU writes each parsed document into a per-method subtree:
//! `{output_dir}/{doc_name}/{parse_method}/`, with extracted images under a
//! sibling `images/` directory. [`prepare_output_dirs`] creates that structure
//! and hands back the paths.

use std::path::{Path, PathBuf};

use crate::error::Result;

/// The concrete output directories for one parsed document.
///
/// Produced by [`prepare_output_dirs`]. Both directories are guaranteed to exist
/// once this value has been returned.
#[derive(Debug, Clone)]
pub struct OutputLayout {
    /// The parse-method directory: `{output_dir}/{doc_name}/{parse_method}`.
    pub base: PathBuf,
    /// The `images/` subdirectory used for extracted figures and tables.
    pub images: PathBuf,
}

/// Create `{output_dir}/{doc_name}/{parse_method}` and its `images/`
/// subdirectory, returning both as an [`OutputLayout`].
///
/// Mirrors the Python `prepare_env(output_dir, doc_name, parse_method)` helper.
/// Existing directories are left untouched.
pub fn prepare_output_dirs(
    output_dir: impl AsRef<Path>,
    doc_name: &str,
    parse_method: &str,
) -> Result<OutputLayout> {
    let base = output_dir.as_ref().join(doc_name).join(parse_method);
    let images = base.join("images");
    std::fs::create_dir_all(&images)?;
    Ok(OutputLayout { base, images })
}
