//! A minimal HTTP server exposing the parser over a single POST route.
//!
//! Deliberately tiny: one shared `Box<dyn Backend>` behind an `Arc`, one route
//! (`POST /parse`) that takes raw PDF bytes and returns the rendered Markdown.
//! It is a thin convenience wrapper, not a production service — no auth, no
//! streaming, no image extraction. The core value is the CLI; this exists so a
//! caller can hit the parser over HTTP without shelling out.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use mineru_render::{render_markdown, MakeMode};
use mineru_types::{Backend, DocInput, ParseOptions};

/// Image-reference subdirectory used when rendering (see [`crate::run`]).
const IMAGE_DIR: &str = "images";

/// Shared server state: the parsing engine, behind an `Arc` for cheap cloning
/// into every request.
#[derive(Clone)]
struct AppState {
    backend: Arc<dyn Backend>,
}

/// Runs the server until the process is terminated.
///
/// Binds `addr` and serves one `POST /parse` route (plus a `GET /health`) backed
/// by `backend`. Every request uses default [`ParseOptions`] and multimodal
/// Markdown output.
///
/// # Errors
/// Returns an error if the address cannot be bound or the server loop fails.
pub async fn serve(addr: &str, backend: Box<dyn Backend>) -> anyhow::Result<()> {
    let state = AppState {
        backend: Arc::from(backend),
    };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/parse", post(parse_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(addr = %addr, "mineru server listening (POST /parse with PDF bytes)");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Handles `POST /parse`: raw PDF bytes in, rendered Markdown out.
async fn parse_handler(State(state): State<AppState>, body: Bytes) -> Response {
    let opts = ParseOptions::default();
    match state
        .backend
        .analyze(DocInput::new(body.to_vec()), &opts)
        .await
    {
        Ok(doc) => {
            let md = render_markdown(&doc, MakeMode::MmMarkdown, IMAGE_DIR);
            (StatusCode::OK, md).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "parse request failed");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("parse failed: {e}")).into_response()
        }
    }
}
