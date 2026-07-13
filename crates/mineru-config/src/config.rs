//! The [`Config`] struct and its load / env-override logic.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::device::Device;
use crate::error::Result;
use crate::model_source::ModelSource;

/// Environment variable pointing at an explicit config-file path.
const ENV_CONFIG_PATH: &str = "MINERU_TOOLS_CONFIG_JSON";
/// Environment variable overriding [`Config::models_dir`].
const ENV_MODELS_DIR: &str = "MINERU_MODELS_DIR";
/// Environment variable overriding [`Config::device`].
const ENV_DEVICE_MODE: &str = "MINERU_DEVICE_MODE";
/// Environment variable overriding [`Config::model_source`].
const ENV_MODEL_SOURCE: &str = "MINERU_MODEL_SOURCE";

/// Default model cache directory.
///
/// The user's main disk is tight, so large model weights live on the Archive
/// volume by default. Override with `models_dir` in the config file or the
/// `MINERU_MODELS_DIR` environment variable.
const DEFAULT_MODELS_DIR: &str = "/Volumes/Archive/mineru/models";

/// User configuration for the MinerU document parser.
///
/// This is the Rust mirror of Python's `mineru.json`. Construct it with
/// [`Config::load`] (which reads a file and applies environment overrides) or
/// [`Config::default`] for the built-in defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The compute device inference runs on.
    pub device: Device,
    /// Where model weights are fetched from.
    pub model_source: ModelSource,
    /// Directory that caches downloaded model weights.
    pub models_dir: PathBuf,
    /// Base URL of an external OpenAI-compatible VLM server, if any.
    pub vlm_server_url: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: Device::default(),
            model_source: ModelSource::default(),
            models_dir: PathBuf::from(DEFAULT_MODELS_DIR),
            vlm_server_url: None,
        }
    }
}

impl Config {
    /// Loads configuration, falling back to defaults for anything unspecified.
    ///
    /// Resolution order:
    /// 1. Read the JSON file at `MINERU_TOOLS_CONFIG_JSON` if that variable is
    ///    set, otherwise `~/.mineru.json`. A missing file yields
    ///    [`Config::default`] rather than an error.
    /// 2. Apply environment-variable overrides (see
    ///    [`Config::apply_env_overrides`]).
    ///
    /// # Errors
    /// Returns an error only if a config file exists but cannot be read or
    /// parsed, or if an override variable holds an unparseable value.
    pub fn load() -> Result<Self> {
        let mut config = match Self::config_path() {
            Some(path) => Self::from_file_or_default(&path)?,
            None => Self::default(),
        };
        config.apply_env_overrides(|key| std::env::var(key).ok())?;
        Ok(config)
    }

    /// Reads and parses a config file, treating a missing file as defaults.
    ///
    /// # Errors
    /// Propagates I/O errors other than "not found", and any JSON parse error.
    pub fn from_file_or_default(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(contents) => Ok(serde_json::from_str(&contents)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Applies environment-variable overrides on top of the current values.
    ///
    /// `get` maps a variable name to its value (or `None` if unset). Taking a
    /// closure keeps this pure and testable without mutating the real process
    /// environment. Recognized variables:
    ///
    /// - `MINERU_MODELS_DIR` → [`Config::models_dir`]
    /// - `MINERU_DEVICE_MODE` → [`Config::device`]
    /// - `MINERU_MODEL_SOURCE` → [`Config::model_source`]
    ///
    /// # Errors
    /// Returns [`crate::Error::InvalidDevice`] or
    /// [`crate::Error::InvalidModelSource`] if a present value fails to parse.
    pub fn apply_env_overrides<F>(&mut self, get: F) -> Result<()>
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(dir) = get(ENV_MODELS_DIR) {
            self.models_dir = PathBuf::from(dir);
        }
        if let Some(device) = get(ENV_DEVICE_MODE) {
            self.device = device.parse()?;
        }
        if let Some(source) = get(ENV_MODEL_SOURCE) {
            self.model_source = source.parse()?;
        }
        Ok(())
    }

    /// Resolves the config-file path: `MINERU_TOOLS_CONFIG_JSON` if set, else
    /// `~/.mineru.json`. Returns `None` when neither is available (e.g. no home
    /// directory), in which case callers should fall back to defaults.
    fn config_path() -> Option<PathBuf> {
        if let Some(explicit) = std::env::var_os(ENV_CONFIG_PATH) {
            return Some(PathBuf::from(explicit));
        }
        dirs::home_dir().map(|home| home.join(".mineru.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_models_dir_is_archive_volume() {
        assert_eq!(
            Config::default().models_dir,
            PathBuf::from("/Volumes/Archive/mineru/models")
        );
    }

    #[test]
    fn default_fields() {
        let c = Config::default();
        assert_eq!(c.device, Device::Cpu);
        assert_eq!(c.model_source, ModelSource::HuggingFace);
        assert_eq!(c.vlm_server_url, None);
    }

    #[test]
    fn models_dir_env_override_applies() {
        let mut c = Config::default();
        c.apply_env_overrides(|key| match key {
            ENV_MODELS_DIR => Some("/tmp/models".to_string()),
            _ => None,
        })
        .unwrap();
        assert_eq!(c.models_dir, PathBuf::from("/tmp/models"));
        // Other fields untouched.
        assert_eq!(c.device, Device::Cpu);
    }

    #[test]
    fn device_and_source_env_overrides_apply() {
        let mut c = Config::default();
        c.apply_env_overrides(|key| match key {
            ENV_DEVICE_MODE => Some("cuda:1".to_string()),
            ENV_MODEL_SOURCE => Some("modelscope".to_string()),
            _ => None,
        })
        .unwrap();
        assert_eq!(c.device, Device::Cuda(1));
        assert_eq!(c.model_source, ModelSource::ModelScope);
    }

    #[test]
    fn no_env_vars_leaves_defaults() {
        let mut c = Config::default();
        c.apply_env_overrides(|_| None).unwrap();
        assert_eq!(c, Config::default());
    }

    #[test]
    fn invalid_device_override_errors() {
        let mut c = Config::default();
        let err = c.apply_env_overrides(|key| match key {
            ENV_DEVICE_MODE => Some("quantum".to_string()),
            _ => None,
        });
        assert!(err.is_err());
    }

    #[test]
    fn missing_file_yields_defaults() {
        let path = Path::new("/nonexistent/does/not/exist.json");
        assert_eq!(Config::from_file_or_default(path).unwrap(), Config::default());
    }

    #[test]
    fn partial_json_fills_defaults() {
        // Only device specified; everything else should default.
        let json = r#"{ "device": "mps" }"#;
        let c: Config = serde_json::from_str(json).unwrap();
        assert_eq!(c.device, Device::Mps);
        assert_eq!(c.models_dir, PathBuf::from("/Volumes/Archive/mineru/models"));
        assert_eq!(c.model_source, ModelSource::HuggingFace);
    }

    #[test]
    fn full_roundtrip() {
        let c = Config {
            device: Device::Cuda(2),
            model_source: ModelSource::Local(PathBuf::from("/w")),
            models_dir: PathBuf::from("/cache"),
            vlm_server_url: Some("http://localhost:8000/v1".to_string()),
        };
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<Config>(&json).unwrap(), c);
    }
}
