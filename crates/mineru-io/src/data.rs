//! The [`DataReader`] / [`DataWriter`] abstraction and its local-filesystem
//! implementations.
//!
//! Both traits are small and object-safe so callers can hold a
//! `Box<dyn DataWriter>` without caring whether bytes land on the local disk or,
//! in a future backend, somewhere else. All paths passed to these methods are
//! *relative* to the backend's configured base directory.

use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};

/// A sink for named byte payloads, keyed by a path relative to some base.
///
/// Object-safe: the default [`write_string`](DataWriter::write_string) method
/// takes no generic parameters, so `dyn DataWriter` is usable.
pub trait DataWriter {
    /// Write `bytes` to `rel_path` (relative to the backend base), replacing any
    /// existing contents. Implementations create parent directories as needed.
    fn write(&self, rel_path: &str, bytes: &[u8]) -> Result<()>;

    /// Write a UTF-8 string to `rel_path`. Defaults to [`write`](DataWriter::write)
    /// over the string's bytes.
    fn write_string(&self, rel_path: &str, s: &str) -> Result<()> {
        self.write(rel_path, s.as_bytes())
    }
}

/// A source of named byte payloads, keyed by a path relative to some base.
pub trait DataReader {
    /// Read the full contents of `rel_path` (relative to the backend base).
    fn read(&self, rel_path: &str) -> Result<Vec<u8>>;
}

/// Join `rel_path` under `base`, rejecting any attempt to escape the base
/// directory.
///
/// A path is rejected (with [`Error::PathEscape`]) if it is absolute, contains a
/// root/prefix component, or contains a parent-dir (`..`) component. Plain
/// `.` components are ignored. This is a purely lexical check that requires no
/// filesystem access, so it works for writes to paths that do not yet exist.
fn safe_join(base: &Path, rel_path: &str) -> Result<PathBuf> {
    let rel = Path::new(rel_path);
    let mut out = base.to_path_buf();
    for component in rel.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(Error::PathEscape(rel_path.to_string()));
            }
        }
    }
    Ok(out)
}

/// A [`DataWriter`] that writes files beneath a fixed base directory.
#[derive(Debug, Clone)]
pub struct LocalFsWriter {
    base: PathBuf,
}

impl LocalFsWriter {
    /// Create a writer rooted at `base`. The directory need not exist yet;
    /// parent directories are created on each [`write`](DataWriter::write).
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// The base directory this writer is rooted at.
    pub fn base(&self) -> &Path {
        &self.base
    }
}

impl DataWriter for LocalFsWriter {
    fn write(&self, rel_path: &str, bytes: &[u8]) -> Result<()> {
        let path = safe_join(&self.base, rel_path)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, bytes)?;
        Ok(())
    }
}

/// A [`DataReader`] that reads files from beneath a fixed base directory.
#[derive(Debug, Clone)]
pub struct LocalFsReader {
    base: PathBuf,
}

impl LocalFsReader {
    /// Create a reader rooted at `base`.
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// The base directory this reader is rooted at.
    pub fn base(&self) -> &Path {
        &self.base
    }
}

impl DataReader for LocalFsReader {
    fn read(&self, rel_path: &str) -> Result<Vec<u8>> {
        let path = safe_join(&self.base, rel_path)?;
        let bytes = std::fs::read(&path)?;
        Ok(bytes)
    }
}
