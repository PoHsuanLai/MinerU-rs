//! User configuration for the MinerU document parser.
//!
//! This crate is the Rust mirror of Python's `mineru.json`. It defines a single
//! [`Config`] struct describing which compute [`Device`] to run on, where model
//! weights come from ([`ModelSource`]), the cache directory for those weights,
//! and an optional external VLM server URL.
//!
//! Load configuration with [`Config::load`], which reads a JSON file (from
//! `MINERU_TOOLS_CONFIG_JSON` or `~/.mineru.json`), fills any unspecified fields
//! from [`Config::default`], and then applies environment-variable overrides. A
//! missing config file is not an error — it simply yields the defaults.
//!
//! Device and model-source values are closed enums with [`std::str::FromStr`]
//! and [`std::fmt::Display`] implementations, so nothing stringly-typed leaks
//! into the rest of the workspace.

pub mod config;
pub mod device;
pub mod download;
pub mod error;
pub mod model_source;

pub use config::Config;
pub use download::{download_missing_models, DEFAULT_MODELS_BASE, REQUIRED_MODEL_FILES};
pub use device::Device;
pub use error::{Error, Result};
pub use model_source::ModelSource;
