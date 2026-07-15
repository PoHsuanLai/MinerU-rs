# MinerU-rs

A Rust port of [MinerU](https://github.com/opendatalab/MinerU) — a document-parsing
engine that turns PDFs into structured Markdown and JSON. The deep-learning models are
reimplemented on the [Burn](https://burn.dev) framework (no ONNX Runtime, no Python at
inference time); non-ML pieces reuse mature Rust crates.

The workspace is a set of **thin, independently-importable crates** (one per model,
one per concern) plus a single `mineru` binary. A downstream Rust project can depend on
just the umbrella `mineru` crate (feature-gated) or on an individual model crate.

> **Status:** offline pipeline is functional end-to-end. Every neural model is
> numerically parity-checked against a PyTorch/ONNX reference (see each crate's
> `tests/`). This is a research port — treat output as verified per-model, not yet
> hardened for production.

## Backends

| Backend | Flag | What it needs |
|---|---|---|
| **pipeline** | `-b pipeline` | Fully local Burn models on disk. No network at inference (weights fetched once). |
| **vlm** | `-b vlm` | An external OpenAI-compatible VLM server (e.g. Qwen2-VL via vLLM / mistral.rs). |
| **hybrid** | `-b hybrid` | Both of the above: local layout models detect the regions, the VLM extracts each one, pipeline OCR post-fills. See `--effort`. |

The hybrid backend needs **both** halves — a models directory *and* a reachable VLM
server:

```bash
mineru paper.pdf -o out -b hybrid --vlm-url http://127.0.0.1:8000/v1 --vlm-model MinerU2.5
```

`--effort medium` (the default) has the local layout model detect each region and the
VLM extract it — one VLM call per region, no VLM layout pass, image/chart analysis off.
`--effort high` has the VLM run its own layout pass as well, using the local layout only
for title-splitting and OCR sidecars. Hybrid's local models always run on the CPU.

Office formats (docx/pptx/xlsx) are deferred (a future `mineru-office` crate).

## Quick start (pipeline backend, fully local)

```sh
# 1. Build the CLI. No model files are needed to build.
cargo build --release -p mineru --bin mineru

# 2. Run. That's it.
./target/release/mineru paper.pdf -o out
# writes out/paper.md, out/paper_content_list.json, and out/images/
```

On first run the model weights (~1 GB) auto-download from the upstream Hugging Face
repo and are cached; later runs do no network I/O. To keep them somewhere specific —
a large-storage volume, say — set `MINERU_MODELS_DIR` first:

```sh
export MINERU_MODELS_DIR=/path/to/PDF-Extract-Kit-1.0/models
```

The GPU is used automatically when a usable adapter is present, falling back to CPU;
`--cpu` forces the exact, reproducible CPU path. See the table-recognition note under
[Model directory layout](#model-directory-layout) for the one stage that still needs
manual weights.

### Native PDFium library

PDF rasterization and native-text extraction use the PDFium native library, loaded at
runtime (none is bundled). Resolution order:

1. `MINERU_PDFIUM_LIB_PATH` — an explicit path to the library.
2. Common system locations (`/opt/homebrew/lib`, `/usr/local/lib`) and the platform default.
3. Auto-download of a matching prebuilt binary, cached under the model/cache directory.

The resolved path (and any download) is logged at info level. To use a specific build,
set `MINERU_PDFIUM_LIB_PATH=/path/to/libpdfium.dylib` (`.so` on Linux, `pdfium.dll` on
Windows).

## Model directory layout

Weights **auto-download on first run** from the upstream
[`opendatalab/PDF-Extract-Kit-1.0`](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0)
Hugging Face repo (pinned to a commit, ~1 GB), landing in `MINERU_MODELS_DIR` — or a
per-user cache dir when that is unset. Nothing needs provisioning by hand; files already
present are never re-fetched, so a fully-populated dir does no network I/O.

Set `MINERU_MODELS_DIR` to control where they land (e.g. a large-storage volume). It
should point at the `models/` directory of the opendatalab release:

```
models/
  Layout/PP-DocLayoutV2/model.safetensors          # layout detector (RT-DETR)
  OCR/paddleocr_torch/
    ch_PP-OCRv6_small_det_infer.safetensors         # text-line detector (DBNet)
    ch_PP-OCRv6_small_rec_infer.safetensors         # text recognizer (SVTR+CTC)
  MFR/unimernet_hf_small_2503/                       # formula recognizer (UniMerNet)
    model.safetensors
    tokenizer.json
  TabRec/SlanetPlus/slanet-plus.onnx                # wireless-table structure (SLANet)
```

The OCR character dictionary is embedded in the binary.

> **Table recognition is currently unavailable on a fresh install.** Both table stages
> need weights converted into Burn's formats — the LCNet classifier and UNet segmenter
> as `.bpk`, SLANet as a `.safetensors` sibling of its `.onnx` — and neither exists in
> any upstream repo. They are derivatives of PDF-Extract-Kit-1.0, which is AGPL-3.0 at
> the repo level, so re-hosting them is on hold pending a licensing answer from
> opendatalab. Everything else (layout, OCR, formula, image extraction, and the VLM and
> hybrid backends) works out of the box. Point `MINERU_TABLE_WEIGHTS_BASE` at your own
> host, or drop the files into `<MINERU_MODELS_DIR>/table-weights-v1/`, to enable them
> meanwhile; a table-model failure warns and emits no table rather than failing the run.

## Environment variables

| Variable | Purpose |
|---|---|
| `MINERU_MODELS_DIR` | Root of the local model weights (pipeline backend), and where auto-download writes them. |
| `MINERU_MODELS_BASE` | Override the base URL pipeline weights auto-download from (default: the pinned upstream Hugging Face revision). |
| `MINERU_PDFIUM_LIB_PATH` | Explicit path to the PDFium native library. |
| `MINERU_TABLE_WEIGHTS_BASE` | Override the base URL for the table `.bpk` weight download (see the note above — the default is currently unpublished). |
| `MINERU_PDFIUM_DOWNLOAD_BASE` | Override the base URL for the PDFium auto-download. |
| `MINERU_VLM_URL` | Base URL of the OpenAI-compatible VLM server (vlm backend). |
| `MINERU_TOOLS_CONFIG_JSON` | Path to a JSON config file (falls back to `~/.mineru.json`). |

Per-model weight overrides (`MINERU_LAYOUT_WEIGHTS`, `MINERU_OCR_DET_WEIGHTS`,
`MINERU_OCR_REC_WEIGHTS`, `MINERU_FORMULA_WEIGHTS`, `MINERU_FORMULA_MODEL_DIR`) point a
single stage at a specific file, bypassing the `MINERU_MODELS_DIR` layout.

## CLI

```
mineru [OPTIONS] <PDF>

  <PDF>                    Input PDF path
  -o, --output <OUTPUT>    Output directory (default: output)
  -b, --backend <BACKEND>  pipeline | vlm | hybrid  (default: pipeline)
      --effort <EFFORT>    medium | high — which layout source drives extraction
                           (hybrid only; default medium)
      --cpu                Force CPU (default: use the GPU when one is usable)
      --lang <LANG>        OCR language hint (e.g. ch, en); omit to auto-detect
      --no-formula         Disable formula recognition
      --no-table           Disable table recognition
      --no-images          Drop images/charts (text-only Markdown, no crops written)
      --pages <PAGES>      Page range START or START:END (0-based, END exclusive)
      --vlm-url <URL>      VLM server base URL      (vlm / hybrid)
      --vlm-model <NAME>   VLM served model name    (vlm / hybrid)
      --debug-output       Also write <stem>_document.json (the full parsed tree)
  -v, --verbose            Debug-level logging (including noisy dependencies)
      --config <CONFIG>    Path to a JSON config file
```

Logs go to stderr at `info`. GPU-backend dependencies (`cubecl_wgpu`, `wgpu_core`,
`wgpu_hal`, `naga`) are pinned to `warn` by default — they log the adapter and every
supported device feature on each run, which `mineru` already reports in one line.
`RUST_LOG` overrides the default entirely (`RUST_LOG=cubecl_wgpu=info` brings that
back), and `-v` raises everything to `debug` with no quieting.

## Using the crates as a library

The umbrella `mineru` crate re-exports the workspace behind cargo features, so you pull
only what you need:

```toml
[dependencies]
# Just the layout model:
mineru = { git = "https://github.com/PoHsuanLai/MinerU-rs", default-features = false, features = ["layout"] }
```

```rust
use mineru::layout::LayoutModel;   // re-exported from mineru-layout
use mineru::types::BBox;           // shared geometry from mineru-types
```

Features: `pipeline`, `vlm`, `hybrid`, and per-model `ocr` / `layout` / `table` /
`formula` / `burn-common`. `cli` (default) builds the binary. Each model crate
(`mineru-layout`, `mineru-ocr-rec`, …) is also publishable/importable on its own.

## Workspace layout

```
mineru-types           domain model (Document/Block/Span enums) + the Backend trait
mineru-config          serde config (device, model source, paths)
mineru-io              filesystem I/O + hf-hub download helper
mineru-pdf             PDFium rasterize + native-text extraction
mineru-render          blocks → Markdown + content_list JSON
mineru-burn-common     shared Burn harness: weight loading, NN blocks, geometry
mineru-ocr-det         DBNet text detection            (Burn)
mineru-ocr-rec         SVTR + CTC text recognition     (Burn)
mineru-layout          RT-DETR layout detection        (Burn)
mineru-formula         UniMerNet formula recognition   (Burn)
mineru-table           LCNet cls + UNet seg + SLANet    (Burn)
mineru-vlm-client      OpenAI-compatible VLM client (no Burn)
mineru-backend-*       pipeline / vlm / hybrid, each impl Backend
mineru                 umbrella library + CLI binary
```

## Development

```sh
cargo build --workspace
cargo test  --workspace                     # fast, offline; heavy tests are #[ignore]d
cargo clippy --workspace --all-targets

# Per-model numeric parity gates (need weights on disk; slow CPU forward):
MINERU_MODELS_DIR=... cargo test -p mineru-table --release --test slanet_real -- --ignored
```

## License

See [`LICENSE`](LICENSE).

## Acknowledgements

Ports [opendatalab/MinerU](https://github.com/opendatalab/MinerU). Model weights are the
original authors'.
