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
    /// Directory holding (or caching) the model weights.
    ///
    /// Resolved from the `MINERU_MODELS_DIR` environment variable or the
    /// `models_dir` config-file key. When neither is set it falls back to a
    /// default cache directory (see [`default_models_dir`]) rather than staying
    /// empty, so a clean machine can auto-download weights into it. An explicit
    /// env/config value always wins over the default.
    pub models_dir: PathBuf,
    /// Base URL of an external OpenAI-compatible VLM server, if any.
    pub vlm_server_url: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: Device::default(),
            model_source: ModelSource::default(),
            // A default cache dir (rather than empty) so a clean machine can
            // auto-download weights into it. Explicit env/config values still win
            // via `apply_env_overrides` / deserialization.
            models_dir: default_models_dir(),
            vlm_server_url: None,
        }
    }
}

/// Resolves the default models cache directory used when neither
/// `MINERU_MODELS_DIR` nor the config file provides one.
///
/// Resolution order:
/// 1. `dirs::cache_dir()` joined with `mineru/models` — the standard per-user
///    cache directory (`~/Library/Caches` on macOS, `$XDG_CACHE_HOME` / `~/.cache`
///    on Linux). This is the only built-in default.
/// 2. `./mineru-models` (cwd-relative) — last resort when no cache dir resolves.
///
/// No machine-specific absolute path is baked into source: a developer who keeps
/// weights on a custom volume points `MINERU_MODELS_DIR` at it, which overrides
/// this default (see [`Config::apply_env_overrides`]).
pub fn default_models_dir() -> PathBuf {
    resolve_default_models_dir(dirs::cache_dir())
}

/// Pure core of [`default_models_dir`], split out so both branches are
/// unit-testable without depending on the host's `dirs` output.
///
/// `cache_dir` is the resolved per-user cache root (from `dirs::cache_dir()`), if
/// any.
fn resolve_default_models_dir(cache_dir: Option<PathBuf>) -> PathBuf {
    match cache_dir {
        Some(dir) => dir.join("mineru").join("models"),
        None => PathBuf::from("./mineru-models"),
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
    fn default_models_dir_is_nonempty_cache_path() {
        // The default is no longer empty: it resolves to a cache dir so a clean
        // machine can auto-download weights into it.
        assert_ne!(Config::default().models_dir, PathBuf::new());
    }

    #[test]
    fn resolve_default_uses_cache_dir() {
        let d = resolve_default_models_dir(Some(PathBuf::from("/some/cache")));
        assert_eq!(d, PathBuf::from("/some/cache/mineru/models"));
    }

    #[test]
    fn resolve_default_falls_back_to_cwd_when_no_cache_dir() {
        let d = resolve_default_models_dir(None);
        assert_eq!(d, PathBuf::from("./mineru-models"));
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
        // Unspecified models_dir falls back to the default cache dir (via
        // `#[serde(default)]` → `Config::default().models_dir`), not empty.
        assert_eq!(c.models_dir, default_models_dir());
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
