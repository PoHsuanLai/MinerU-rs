//! OpenAI-compatible client for the MinerU VLM.
//!
//! Talks to an external Qwen2-VL server (vLLM/SGLang/mistral.rs) over the
//! OpenAI-compatible chat-completions API, sending page images and parsing the
//! model's block output. The parsed blocks are assembled into the canonical
//! [`Document`](mineru_types::Document) tree by [`assemble`].
//!
//! The HTTP/prompt layer (module [`client`]) is kept separate from the
//! model-output-to-document conversion ([`assemble`]) so the wire format can
//! evolve without touching the document mapping.

pub mod assemble;
pub mod client;
pub mod error;
pub mod parse;
pub mod prompts;
pub mod raw;

pub use assemble::assemble_document;
pub use client::{VlmClient, VlmClientConfig};
pub use error::{Error, Result};
pub use parse::parse_layout;
pub use raw::{VlmBlock, VlmPage};
