//! Where model weights are fetched from.

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// The source model weights are downloaded (or loaded) from.
///
/// Mirrors the Python `models-source` option, plus a `Local(path)` escape hatch
/// for weights already present on disk.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// Download from the Hugging Face Hub. The default.
    #[default]
    HuggingFace,
    /// Download from ModelScope (useful in mainland China).
    ModelScope,
    /// Load from a local directory, bypassing any download.
    Local(PathBuf),
}

impl FromStr for ModelSource {
    type Err = Error;

    /// Parses `"huggingface"`, `"modelscope"`, or `"local:<path>"`.
    ///
    /// A few common spellings/aliases are accepted; parsing is case-insensitive
    /// for the two remote sources.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if let Some(path) = trimmed.strip_prefix("local:") {
            return Ok(ModelSource::Local(PathBuf::from(path)));
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "huggingface" | "hugging_face" | "hf" => Ok(ModelSource::HuggingFace),
            "modelscope" | "model_scope" | "ms" => Ok(ModelSource::ModelScope),
            "local" => Ok(ModelSource::Local(PathBuf::new())),
            _ => Err(Error::InvalidModelSource(s.to_string())),
        }
    }
}

impl fmt::Display for ModelSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSource::HuggingFace => f.write_str("huggingface"),
            ModelSource::ModelScope => f.write_str("modelscope"),
            ModelSource::Local(path) => write!(f, "local:{}", path.display()),
        }
    }
}

impl Serialize for ModelSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ModelSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_remote_sources() {
        assert_eq!(
            "huggingface".parse::<ModelSource>().unwrap(),
            ModelSource::HuggingFace
        );
        assert_eq!(
            "ModelScope".parse::<ModelSource>().unwrap(),
            ModelSource::ModelScope
        );
    }

    #[test]
    fn parses_local_path() {
        assert_eq!(
            "local:/opt/models".parse::<ModelSource>().unwrap(),
            ModelSource::Local(PathBuf::from("/opt/models"))
        );
    }

    #[test]
    fn default_is_hugging_face() {
        assert_eq!(ModelSource::default(), ModelSource::HuggingFace);
    }

    #[test]
    fn rejects_unknown() {
        assert!("s3".parse::<ModelSource>().is_err());
    }

    #[test]
    fn serde_roundtrips() {
        for src in [
            ModelSource::HuggingFace,
            ModelSource::ModelScope,
            ModelSource::Local(PathBuf::from("/data/w")),
        ] {
            let json = serde_json::to_string(&src).unwrap();
            assert_eq!(serde_json::from_str::<ModelSource>(&json).unwrap(), src);
        }
    }
}
