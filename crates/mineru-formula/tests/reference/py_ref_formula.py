#!/usr/bin/env python3
"""Self-contained PyTorch reference forward for UniMerNet (unimernet_hf_small_2503).

This reimplements the *exact* forward math of the HuggingFace reference
(`mineru/model/mfr/unimernet/unimernet_hf/`) directly on top of the checkpoint
tensors, WITHOUT importing `transformers` (which is not installed here). Every
op below is a line-for-line translation of:

  - unimer_swin/modeling_unimer_swin.py  (Swin encoder: StemLayer, patch embed,
    ConvEnhance, window attention with relative-position bias, patch merging)
  - unimer_mbart/modeling_unimer_mbart.py (MBart decoder: scaled word embedding,
    learned positions offset by 2, squeeze self/cross attention, LM head)

It loads the SAME `model.safetensors` the Rust crate loads, runs a DETERMINISTIC
input, and dumps for the Rust parity test:

  - input                (input.{bin,shape})        [1, 3, 192, 672] grayscale-repeated
  - swin_embed           patch-embed output         [1, L0, 96]
  - swin_stage_0..3      each Swin stage output      [1, L_i, C_i]
  - encoder_out          == swin_stage_3            [1, L3, 768]
  - decoder_logits       first decode step (BOS)    [1, 1, 50000]

Methodology (mirrors the proven `mineru-ocr-det` template):
  * The image processor targets [H=192, W=672] and produces ONE grayscale channel
    normalised by `(gray - 0.7931*255)/(0.1738*255)`, then repeats it to 3 chans.
  * We build the input tensor DIRECTLY at the target size so resize/crop/pad are
    identity -- isolating model math from resize-interpolation differences. The
    Rust test builds the byte-identical tensor and separately validates its own
    preprocess() path against `input`.
  * The decoder first step feeds ONLY the BOS token (id 0), which is deterministic
    (T=1 -> no causal masking effect); we dump those logits. We do NOT diff a
    sampled sequence (sampling diverges chaotically).

Override the checkpoint dir via FORMULA_WEIGHTS (defaults to the on-disk PEK-1.0
location). Dumps are written next to this script; point the Rust test's
FORMULA_REF_DIR here.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

WEIGHTS_DIR = os.environ.get(
    "FORMULA_WEIGHTS",
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/MFR/unimernet_hf_small_2503",
)
CKPT = os.path.join(WEIGHTS_DIR, "model.safetensors")
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- config (unimernet_hf_small_2503, from config.json) --------------------
IMG_H, IMG_W = 192, 672          # image-processor target [height, width]
EMBED_DIM = 96
DEPTHS = [6, 6, 6, 6]
NUM_HEADS = [3, 6, 12, 24]
WINDOW = 5
MLP_RATIO = 4.0
LN_EPS = 1e-5
D_MODEL = 768
DEC_LAYERS = 8
DEC_HEADS = 16
QK_SQUEEZE = 2
FFN_DIM = 3072
VOCAB = 50000
POS_OFFSET = 2
BOS = 0

NORM_MEAN = 0.7931
NORM_STD = 0.1738


def dump(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(os.path.join(OUT, name + ".bin"), "wb") as f:
        f.write(arr.tobytes())
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in arr.shape))
    print(f"[dump] {name} shape {tuple(arr.shape)}")


# ---- deterministic input ---------------------------------------------------
def make_input():
    """A deterministic [1,3,192,672] pixel tensor.

    We synthesize a grayscale gradient in [0,255], normalise with UniMerNet's
    constants, then repeat to 3 channels (matching generate()'s
    pixel_values.repeat(1,3,1,1)). This is exactly what the Rust test builds.
    """
    yy, xx = np.meshgrid(np.arange(IMG_H), np.arange(IMG_W), indexing="ij")
    # A structured gray pattern spanning the full [0,255] range.
    gray = ((xx * 173 + yy * 149) % 256).astype(np.float32)  # HxW in [0,255]
    normalized = (gray - NORM_MEAN * 255.0) / (NORM_STD * 255.0)  # HxW
    chan = normalized[None, None, :, :]                            # [1,1,H,W]
    inp = np.repeat(chan, 3, axis=1)                              # [1,3,H,W]
    return np.ascontiguousarray(inp, dtype=np.float32)


# ---- weight access ---------------------------------------------------------
class W:
    def __init__(self, sd):
        self.sd = sd

    def __getitem__(self, k):
        return self.sd[k]


def linear(x, w, b, key):
    weight = w[key + ".weight"]
    bias = w.sd.get(key + ".bias")
    return F.linear(x, weight, bias)


def layernorm(x, w, key, eps=LN_EPS):
    return F.layer_norm(x, (x.shape[-1],), w[key + ".weight"], w[key + ".bias"], eps)


# ---- Swin encoder ----------------------------------------------------------
def stem(x, w):
    # projection.conv1 (3->48, k3 s2 p1), norm1.0 = BatchNorm2d(48) frozen, GELU,
    # projection.conv2 (48->96, k3 s2 p1)
    pre = "encoder.embeddings.patch_embeddings.projection"
    x = F.conv2d(x, w[pre + ".conv1.weight"], w[pre + ".conv1.bias"], stride=2, padding=1)
    x = F.batch_norm(
        x,
        w[pre + ".norm1.0.running_mean"],
        w[pre + ".norm1.0.running_var"],
        w[pre + ".norm1.0.weight"],
        w[pre + ".norm1.0.bias"],
        training=False,
        eps=1e-5,
    )
    x = F.gelu(x)
    x = F.conv2d(x, w[pre + ".conv2.weight"], w[pre + ".conv2.bias"], stride=2, padding=1)
    return x  # [B, 96, H/4, W/4]


def patch_embed(pixel_values, w):
    x = stem(pixel_values, w)
    _, dim, h, wid = x.shape
    x = x.flatten(2).transpose(1, 2)  # [B, H*W, dim]
    x = layernorm(x, w, "encoder.embeddings.norm")
    return x, (h, wid)


def relative_position_index(window):
    coords_h = torch.arange(window)
    coords_w = torch.arange(window)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += window - 1
    rel[:, :, 1] += window - 1
    rel[:, :, 0] *= 2 * window - 1
    return rel.sum(-1)  # [N, N]


def window_partition(x, win):
    b, h, wid, c = x.shape
    x = x.view(b, h // win, win, wid // win, win, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win, win, c)


def window_reverse(windows, win, h, wid):
    c = windows.shape[-1]
    x = windows.view(-1, h // win, wid // win, win, win, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h, wid, c)


def conv_enhance(x, w, key, h, wid):
    b, n, c = x.shape
    feat = x.transpose(1, 2).view(b, c, h, wid)
    feat = F.conv2d(feat, w[key + ".proj.weight"], w[key + ".proj.bias"], stride=1, padding=1, groups=c)
    feat = F.gelu(feat)
    feat = feat.flatten(2).transpose(1, 2)
    return x + feat


def window_attention(x, w, key, num_heads, win):
    # x: [num_windows*B, N, C]
    bw, n, dim = x.shape
    head_dim = dim // num_heads
    q = linear(x, w, None, key + ".self.query")
    k = linear(x, w, None, key + ".self.key")
    v = linear(x, w, None, key + ".self.value")

    def heads(t):
        return t.view(bw, n, num_heads, head_dim).permute(0, 2, 1, 3)

    q, k, v = heads(q), heads(k), heads(v)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)

    table = w[key + ".self.relative_position_bias_table"]  # [(2W-1)^2, heads]
    idx = relative_position_index(win).view(-1)
    bias = table[idx].view(win * win, win * win, -1).permute(2, 0, 1).contiguous()
    scores = scores + bias.unsqueeze(0)

    probs = F.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(bw, n, dim)
    return linear(ctx, w, None, key + ".output.dense")


def swin_block(x, w, key, num_heads, h, wid, win_cfg):
    b, _, c = x.shape
    win = min(win_cfg, h, wid)

    x = conv_enhance(x, w, key + ".ce.0", h, wid)
    shortcut = x

    normed = layernorm(x, w, key + ".layernorm_before")
    normed = normed.view(b, h, wid, c)
    pad_b = (win - h % win) % win
    pad_r = (win - wid % win) % win
    if pad_b or pad_r:
        normed = F.pad(normed, (0, 0, 0, pad_r, 0, pad_b))
    hp, wp = h + pad_b, wid + pad_r

    windows = window_partition(normed, win).view(-1, win * win, c)
    attn = window_attention(windows, w, key + ".attention", num_heads, win)
    attn = attn.view(-1, win, win, c)
    attn = window_reverse(attn, win, hp, wp)
    if pad_b or pad_r:
        attn = attn[:, :h, :wid, :].contiguous()
    attn = attn.view(b, h * wid, c)

    x = shortcut + attn
    x = conv_enhance(x, w, key + ".ce.1", h, wid)

    ff = layernorm(x, w, key + ".layernorm_after")
    ff = F.gelu(linear(ff, w, None, key + ".intermediate.dense"))
    ff = linear(ff, w, None, key + ".output.dense")
    return x + ff


def patch_merging(x, w, key, h, wid):
    b, _, c = x.shape
    x = x.view(b, h, wid, c)
    if h % 2 or wid % 2:
        x = F.pad(x, (0, 0, 0, wid % 2, 0, h % 2))
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    merged = torch.cat([x0, x1, x2, x3], -1)
    merged = merged.view(b, -1, 4 * c)
    merged = layernorm(merged, w, key + ".norm")
    return linear(merged, w, None, key + ".reduction")


def swin_encoder(pixel_values, w, dumps):
    x, (h, wid) = patch_embed(pixel_values, w)
    dumps["swin_embed"] = x
    for i in range(4):
        stage_key = f"encoder.encoder.layers.{i}"
        for bi in range(DEPTHS[i]):
            x = swin_block(x, w, f"{stage_key}.blocks.{bi}", NUM_HEADS[i], h, wid, WINDOW)
        if i < 3:
            x = patch_merging(x, w, f"{stage_key}.downsample", h, wid)
            h, wid = (h + 1) // 2, (wid + 1) // 2
        dumps[f"swin_stage_{i}"] = x
    return x


# ---- MBart decoder (first step, BOS) ---------------------------------------
def squeeze_attention(hidden, kv, w, key, causal):
    # squeeze: q/k -> squeeze_dim (384), v/out -> d_model (768)
    b, tgt, _ = hidden.shape
    squeeze_dim = D_MODEL // QK_SQUEEZE
    squeeze_head = squeeze_dim // DEC_HEADS
    head_dim = D_MODEL // DEC_HEADS
    scaling = squeeze_head ** -0.5

    q = linear(hidden, w, None, key + ".q_proj") * scaling
    k = linear(kv, w, None, key + ".k_proj")
    v = linear(kv, w, None, key + ".v_proj")
    src = kv.shape[1]

    def shape_qk(t):
        return t.view(b, -1, DEC_HEADS, squeeze_head).transpose(1, 2)

    def shape_v(t):
        return t.view(b, -1, DEC_HEADS, head_dim).transpose(1, 2)

    q, k, v = shape_qk(q), shape_qk(k), shape_v(v)
    scores = torch.matmul(q, k.transpose(-1, -2))  # [B, heads, tgt, src]
    if causal is not None:
        scores = scores + causal
    probs = F.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, tgt, D_MODEL)
    return linear(ctx, w, None, key + ".out_proj")


def decoder_first_step(encoder_hidden, w):
    ids = torch.tensor([[BOS]], dtype=torch.long)  # [1,1]
    b, t = ids.shape
    embed_scale = math.sqrt(D_MODEL)
    emb = F.embedding(ids, w["decoder.model.decoder.embed_tokens.weight"]) * embed_scale
    pos_ids = torch.arange(t) + POS_OFFSET
    pos = F.embedding(pos_ids.unsqueeze(0), w["decoder.model.decoder.embed_positions.weight"])
    hidden = emb + pos
    hidden = layernorm(hidden, w, "decoder.model.decoder.layernorm_embedding")

    # Causal mask for T=1 is all-zeros (single token attends only to itself).
    causal = torch.zeros(b, 1, t, t)

    for li in range(DEC_LAYERS):
        key = f"decoder.model.decoder.layers.{li}"
        residual = hidden
        x = layernorm(hidden, w, key + ".self_attn_layer_norm")
        x = squeeze_attention(x, x, w, key + ".self_attn", causal)
        hidden = residual + x

        residual = hidden
        x = layernorm(hidden, w, key + ".encoder_attn_layer_norm")
        x = squeeze_attention(x, encoder_hidden, w, key + ".encoder_attn", None)
        hidden = residual + x

        residual = hidden
        x = layernorm(hidden, w, key + ".final_layer_norm")
        x = linear(F.gelu(linear(x, w, None, key + ".fc1")), w, None, key + ".fc2")
        hidden = residual + x

    hidden = layernorm(hidden, w, "decoder.model.decoder.layer_norm")
    logits = F.linear(hidden, w["decoder.lm_head.weight"])
    return logits  # [1, 1, VOCAB]


def main():
    print(f"[ckpt] {CKPT}")
    sd = load_file(CKPT, device="cpu")
    w = W({k: v.float() for k, v in sd.items()})

    inp = make_input()
    dump("input", inp)
    print("[input] min", inp.min(), "max", inp.max(), "mean", inp.mean())

    pixel_values = torch.from_numpy(inp)
    with torch.inference_mode():
        dumps = {}
        enc = swin_encoder(pixel_values, w, dumps)
        for name, t in dumps.items():
            dump(name, t.detach().cpu().numpy())
        dump("encoder_out", enc.detach().cpu().numpy())
        print("[encoder_out] shape", tuple(enc.shape), "mean", float(enc.mean()))

        logits = decoder_first_step(enc, w)
        dump("decoder_logits", logits.detach().cpu().numpy())
        top = int(logits[0, 0].argmax())
        print("[decoder_logits] shape", tuple(logits.shape), "argmax", top,
              "max", float(logits.max()))

    print("[done] dumped to", OUT)


if __name__ == "__main__":
    main()
