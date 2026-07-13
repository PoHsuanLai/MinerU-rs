//! The async HTTP client: talks to an OpenAI-compatible VLM server.
//!
//! Implements the MinerU2.5 two-step flow — a layout pass over the full page,
//! then a per-block content-extraction pass — building [`VlmPage`]s that
//! [`assemble_document`](crate::assemble_document) turns into a document.

use async_openai::config::OpenAIConfig;
use async_openai::types::CreateChatCompletionResponse;
use async_openai::Client;
use base64::Engine;
use image::RgbImage;
use serde_json::json;

use crate::error::{Error, Result};
use crate::parse::parse_layout;
use crate::prompts::{self, Sampling};
use crate::raw::{VlmBlock, VlmPage};

/// Configuration for connecting to the VLM server.
#[derive(Debug, Clone)]
pub struct VlmClientConfig {
    /// OpenAI-compatible base URL, e.g. `http://localhost:30000/v1`.
    pub base_url: String,
    /// The served model name.
    pub model: String,
    /// API key; local servers usually accept any non-empty value.
    pub api_key: String,
    /// Max tokens to request per call.
    pub max_tokens: u32,
}

impl Default for VlmClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:30000/v1".to_owned(),
            model: "MinerU2.5".to_owned(),
            api_key: "EMPTY".to_owned(),
            max_tokens: 4096,
        }
    }
}

/// A client for the MinerU VLM server.
pub struct VlmClient {
    http: Client<OpenAIConfig>,
    config: VlmClientConfig,
}

impl VlmClient {
    /// Builds a client from configuration.
    pub fn new(config: VlmClientConfig) -> Self {
        let openai = OpenAIConfig::new()
            .with_api_base(config.base_url.clone())
            .with_api_key(config.api_key.clone());
        Self {
            http: Client::with_config(openai),
            config,
        }
    }

    /// Runs the full two-step extraction on one rasterized page.
    ///
    /// Step 1 detects the layout over the whole page; step 2 fills each block's
    /// content from a crop. `image_analysis` enables content extraction for
    /// image/chart blocks (off by default, matching the Python).
    pub async fn extract_page(&self, page: &RgbImage, image_analysis: bool) -> Result<VlmPage> {
        let (w, h) = (page.width() as f32, page.height() as f32);

        // Step 1: layout over the full page.
        let layout_text = self
            .complete(page, prompts::LAYOUT, Sampling::default())
            .await?;
        let mut blocks = parse_layout(&layout_text);

        // Step 2: per-block content extraction.
        for block in &mut blocks {
            if skip_extraction(&block.label, image_analysis) {
                continue;
            }
            let crop = crop_block(page, block);
            let prompt = prompts::extraction_prompt(&block.label);
            let sampling = Sampling::for_extraction(&block.label);
            match self.complete(&crop, prompt, sampling).await {
                Ok(content) => block.content = Some(strip_end_token(&content)),
                Err(e) => {
                    tracing::warn!("extraction failed for {}: {e}", block.label);
                }
            }
        }

        Ok(VlmPage {
            width: w,
            height: h,
            blocks,
        })
    }

    /// Sends one image + prompt and returns the model's text output.
    ///
    /// Built as a raw JSON request (via async-openai's `byot` path) so the
    /// vLLM/SGLang sampling extensions (`top_k`, `repetition_penalty`,
    /// `vllm_xargs`, …) — which the typed OpenAI schema omits — go on the wire
    /// verbatim. The image content part precedes the text, matching the
    /// reference client.
    async fn complete(&self, image: &RgbImage, prompt: &str, sampling: Sampling) -> Result<String> {
        let data_url = png_data_url(image)?;

        let request = json!({
            "model": self.config.model,
            "messages": [
                { "role": "system", "content": prompts::SYSTEM },
                { "role": "user", "content": [
                    { "type": "image_url", "image_url": { "url": data_url } },
                    { "type": "text", "text": prompt },
                ]},
            ],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "presence_penalty": sampling.presence_penalty,
            "frequency_penalty": sampling.frequency_penalty,
            "max_tokens": self.config.max_tokens,
            // vLLM/SGLang extensions absent from the OpenAI schema.
            "top_k": sampling.top_k,
            "repetition_penalty": sampling.repetition_penalty,
            "skip_special_tokens": false,
            "vllm_xargs": { "no_repeat_ngram_size": sampling.no_repeat_ngram_size },
        });

        let response: CreateChatCompletionResponse = self
            .http
            .chat()
            .create_byot(request)
            .await
            .map_err(|e| Error::Request(e.to_string()))?;

        let text = response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| Error::Parse("empty completion".to_owned()))?;
        Ok(text)
    }
}

/// Whether a block is skipped in the extraction pass.
fn skip_extraction(label: &str, image_analysis: bool) -> bool {
    match label {
        // Image/chart content is only extracted when explicitly enabled.
        "image" | "chart" | "image_block" => !image_analysis,
        // These carry no extractable content of their own.
        "unknown" => true,
        _ => false,
    }
}

/// Crops the page to a block's (normalized) box. The box is denormalized against
/// the page's own pixel size here, since extraction runs on the crop.
fn crop_block(page: &RgbImage, block: &VlmBlock) -> RgbImage {
    let (w, h) = (page.width() as f32, page.height() as f32);
    let [x0, y0, x1, y1] = block.bbox;
    let px0 = (x0 * w).round().clamp(0.0, w) as u32;
    let py0 = (y0 * h).round().clamp(0.0, h) as u32;
    let px1 = (x1 * w).round().clamp(0.0, w) as u32;
    let py1 = (y1 * h).round().clamp(0.0, h) as u32;
    let cw = px1.saturating_sub(px0).max(1);
    let ch = py1.saturating_sub(py0).max(1);
    image::imageops::crop_imm(page, px0, py0, cw, ch).to_image()
}

/// Encodes an image as a `data:image/png;base64,...` URL.
fn png_data_url(image: &RgbImage) -> Result<String> {
    let mut buf = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| Error::ImageEncode(e.to_string()))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
    Ok(format!("data:image/png;base64,{b64}"))
}

/// Strips the VLM end token from a response, if present.
fn strip_end_token(text: &str) -> String {
    let end = std::env::var("MINERU_VLM_END_TOKEN").unwrap_or_else(|_| "<|im_end|>".to_owned());
    text.trim().trim_end_matches(&end).trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_images_unless_enabled() {
        assert!(skip_extraction("image", false));
        assert!(!skip_extraction("image", true));
        assert!(!skip_extraction("text", false));
    }

    #[test]
    fn strips_end_token() {
        assert_eq!(strip_end_token("hello<|im_end|>"), "hello");
        assert_eq!(strip_end_token("  world  "), "world");
    }

    #[test]
    fn crop_clamps_to_bounds() {
        let img = RgbImage::new(100, 100);
        let block = VlmBlock {
            bbox: [0.1, 0.1, 0.5, 0.5],
            label: "text".to_owned(),
            content: None,
            angle: 0,
            sub_type: None,
        };
        let crop = crop_block(&img, &block);
        assert_eq!((crop.width(), crop.height()), (40, 40));
    }
}
