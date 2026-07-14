#!/usr/bin/env python3
"""Python reference forward for PP-OCRv6 small rec, for Rust parity.

Builds the SVTR_LCNet rec model (PP-LCNetV4 backbone + LightSVTR neck + CTC head),
loads the SAME safetensors checkpoint used by the Rust `mineru-ocr-rec` crate, runs
a DETERMINISTIC synthetic 320x48 RGB crop through preprocessing that matches the
Rust `TextRecognizer::preprocess` EXACTLY (BGR channel order, x/127.5 - 1, right
zero-pad), forwards it, and dumps every intermediate stage as flat little-endian
f32 `.bin` + `.shape`:

  - input                 [1, 3, 48, 320]   (the preprocessed BGR tensor)
  - backbone_0..3         the four PPLCNetV4Block stage outputs
  - backbone_pooled       avg_pool2d([3,2]) height-pooled feature  [1, 384, 1, W']
  - neck                  LightSVTR neck output [1, 120, 1, W']
  - logits                raw CTC logits [1, T, 18710]

The crop is 48 high (== rec image_height) and 320 wide (== rec image_width base),
so `resize_norm_img` is an IDENTITY resize (resized_w == imgW == 320, no cv2
interpolation, no padding). This isolates conv/BN/attention/CTC-head math from
resize-filter differences, which is the point of THIS parity target.

Override the checkpoint with REC_WEIGHTS; dumps are written next to this script.
"""
import os
import sys

import numpy as np
import torch

PYOCR = "/Users/pohsuanlai/Documents/mineru/mineru/mineru/model/utils"
sys.path.insert(0, PYOCR)

# Stub out heavy transitive deps pulled in by build_head's eager imports
# (rec_ppformulanet_head -> mineru.utils.config_reader -> loguru). We only need
# the rec CTC branch, so provide a minimal fake module so the import doesn't fail.
import types as _types

_fake = _types.ModuleType("mineru.utils.config_reader")
_fake.get_device = lambda *a, **k: "cpu"
sys.modules.setdefault("mineru", _types.ModuleType("mineru"))
sys.modules.setdefault("mineru.utils", _types.ModuleType("mineru.utils"))
sys.modules["mineru.utils.config_reader"] = _fake

from pytorchocr.modeling.architectures.base_model import BaseModel

# Checkpoint path: override with REC_WEIGHTS; defaults to the on-disk PEK-1.0 location.
CKPT = os.environ.get(
    "REC_WEIGHTS",
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/OCR/paddleocr_torch/ch_PP-OCRv6_small_rec_infer.safetensors",
)
# Reference tensors are written next to this script; point the Rust parity test's
# REC_REF_DIR at this directory.
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- arch config for ch_PP-OCRv6_small_rec_infer (from arch_config.yaml) ----
config = {
    "model_type": "rec",
    "algorithm": "SVTR_LCNet",
    "Transform": None,
    "Backbone": {"name": "PPLCNetV4", "model_size": "small"},
    "Head": {
        "name": "MultiHead",
        "out_channels_list": {"CTCLabelDecode": 18710},
        "head_list": [
            {
                "CTCHead": {
                    "Neck": {
                        "name": "lightsvtr",
                        "dims": 120,
                        "depth": 2,
                        "mlp_ratio": 2.0,
                        "local_kernel": 7,
                    },
                    "Head": {"fc_decay": 0.00001},
                }
            },
            {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
        ],
    },
    "in_channels": 3,
}


def build_model():
    import copy

    model = BaseModel(copy.deepcopy(config))
    model.eval()
    return model


def load_ckpt(model):
    from safetensors.torch import load_file

    sd = load_file(CKPT, device="cpu")
    # Strip a leading "model." only where it wraps the backbone; the checkpoint keys
    # are model.backbone.* and head.* which BaseModel exposes as backbone.* / head.*.
    if any(k.startswith("model.") for k in sd):
        sd = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in sd.items()
        }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  MISSING (first 10):", missing[:10])
    if unexpected:
        print("  UNEXPECTED (first 10):", unexpected[:10])
    return model


# ---- deterministic synthetic crop: 320x48 RGB gradient/pattern ----
def make_image(h=48, w=320):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = (xx * 255 // (w - 1)).astype(np.uint8)
    g = (yy * 255 // (h - 1)).astype(np.uint8)
    b = (((xx + yy) * 255 // (h + w - 2))).astype(np.uint8)
    img = np.stack([r, g, b], axis=2)  # HWC, RGB, uint8
    return img


# ---- preprocessing matching Rust TextRecognizer::preprocess EXACTLY ----
# BGR channel order, x/127.5 - 1, CHW, NCHW. The crop is already 48x320 so resize is
# identity (resized_w == canvas_w == 320): no cv2 interpolation, no padding.
def preprocess(img_rgb_hwc):
    x = img_rgb_hwc.astype(np.float32)  # HWC RGB
    bgr = x[:, :, ::-1]  # HWC BGR
    bgr = bgr / 127.5 - 1.0
    bgr = np.transpose(bgr, (2, 0, 1))  # CHW (BGR)
    bgr = bgr[None, ...]  # NCHW
    return np.ascontiguousarray(bgr, dtype=np.float32)


def dump_bin(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(os.path.join(OUT, name + ".bin"), "wb") as f:
        f.write(arr.tobytes())
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in arr.shape))
    print(f"[{name}] shape {tuple(arr.shape)}")


def main():
    model = build_model()
    model = load_ckpt(model)

    img = make_image()
    inp = preprocess(img)
    print("[input] shape", inp.shape, "min", inp.min(), "max", inp.max(), "mean", inp.mean())
    dump_bin("input", inp)

    t = torch.from_numpy(inp)
    with torch.inference_mode():
        # ---- backbone: run the encoder to grab each stage, then the height pool ----
        backbone = model.backbone
        feats = backbone.encoder(t)  # list of 4 stage feature maps
        for i, f in enumerate(feats):
            dump_bin(f"backbone_{i}", f.detach().cpu().numpy())
        last = feats[-1]
        pooled = torch.nn.functional.avg_pool2d(last, [3, 2])
        dump_bin("backbone_pooled", pooled.detach().cpu().numpy())

        # ---- head (MultiHead, lightsvtr CTC branch) ----
        head = model.head
        neck = head.encoder(pooled)  # LightSVTR neck, [1, 120, 1, W]
        dump_bin("neck", neck.detach().cpu().numpy())
        ctc = neck.squeeze(dim=2).permute(0, 2, 1)  # [1, W, 120]
        logits = head.head(ctc)  # raw CTC logits [1, T, 18710]
        logits_np = logits.detach().cpu().numpy()
        dump_bin("logits", logits_np)
        print(
            "[logits] min", float(logits_np.min()),
            "max", float(logits_np.max()),
            "mean", float(logits_np.mean()),
        )

    print("[done] dumped .bin to", OUT)


if __name__ == "__main__":
    main()
