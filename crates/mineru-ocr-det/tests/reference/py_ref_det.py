#!/usr/bin/env python3
"""Python reference forward for PP-OCRv6 small det, for Rust parity.

Builds the DBNet det model, loads the SAME safetensors checkpoint used by the
Rust crate, runs a DETERMINISTIC synthetic 320x320 RGB image through
preprocessing that matches the Rust `mineru-burn-common::preprocess::Preprocess`
EXACTLY (RGB order, /255, ImageNet mean/std), forwards it, and dumps:

  - the preprocessed input tensor   (input.npy)   [1,3,320,320]
  - backbone_out feature maps        (backbone_*.npy) list of 4
  - neck_out                         (neck.npy)
  - final maps (sigmoid prob map)    (maps.npy)     [1,1,H,W]

Input is a multiple of 32 in both dims so DetResizeForTest is an identity resize
(ratio=1, no cv2 interpolation) -- this isolates conv/BN math from resize filter
differences, which is the point of THIS parity target.
"""
import os
import sys
import numpy as np
import torch

PYOCR = "/Users/pohsuanlai/Documents/mineru/mineru/mineru/model/utils"
sys.path.insert(0, PYOCR)

# Stub out heavy transitive deps pulled in by build_head's eager imports
# (rec_ppformulanet_head -> mineru.utils.config_reader -> loguru). We only need
# the det DBHead, so provide a minimal fake module so the import doesn't fail.
import types as _types
_fake = _types.ModuleType("mineru.utils.config_reader")
_fake.get_device = lambda *a, **k: "cpu"
sys.modules.setdefault("mineru", _types.ModuleType("mineru"))
sys.modules.setdefault("mineru.utils", _types.ModuleType("mineru.utils"))
sys.modules["mineru.utils.config_reader"] = _fake

from pytorchocr.modeling.architectures.base_model import BaseModel

# Checkpoint path: override with DET_WEIGHTS; defaults to the on-disk PEK-1.0 location.
CKPT = os.environ.get(
    "DET_WEIGHTS",
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/OCR/paddleocr_torch/ch_PP-OCRv6_small_det_infer.safetensors",
)
# Reference tensors are written next to this script; point the Rust parity test's
# DET_REF_DIR at this directory.
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- arch config for ch_PP-OCRv6_small_det_infer (from arch_config.yaml) ----
config = {
    "model_type": "det",
    "algorithm": "DB",
    "Transform": None,
    "Backbone": {"name": "PPLCNetV4", "det": True, "model_size": "small"},
    "Neck": {"name": "RepLKFPN", "out_channels": 96, "dilated_kernel_size": 7,
             "shortcut": True},
    "Head": {"name": "DBHead", "k": 50, "mode": "ppocrv6", "fix_nan": True,
             "aux_in_channels": 96, "kernel_list": [3, 2, 2]},
    "return_all_feats": True,  # so we can grab backbone_out + neck_out
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
    # strip leading "model." (backbone/neck are under model.*, head is head.*)
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):] if k.startswith("model.") else k: v
              for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  MISSING (first 10):", missing[:10])
    if unexpected:
        print("  UNEXPECTED (first 10):", unexpected[:10])
    return model

# ---- deterministic synthetic image: 320x320 RGB gradient/pattern ----
def make_image(h=320, w=320):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = (xx * 255 // (w - 1)).astype(np.uint8)
    g = (yy * 255 // (h - 1)).astype(np.uint8)
    b = (((xx + yy) * 255 // (h + w - 2))).astype(np.uint8)
    img = np.stack([r, g, b], axis=2)  # HWC, RGB, uint8
    return img

# ---- preprocessing matching Rust Preprocess exactly ----
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_rgb_hwc):
    # img is uint8 HWC RGB. Rust: (px/255 - mean)/std, per RGB channel, CHW.
    x = img_rgb_hwc.astype(np.float32) / 255.0  # HWC
    x = (x - IMAGENET_MEAN) / IMAGENET_STD      # broadcast over channels
    x = np.transpose(x, (2, 0, 1))              # CHW (RGB)
    x = x[None, ...]                            # NCHW
    return np.ascontiguousarray(x, dtype=np.float32)

def main():
    model = build_model()
    model = load_ckpt(model)

    img = make_image()
    inp = preprocess(img)
    np.save(os.path.join(OUT, "input.npy"), inp)
    print("[input] shape", inp.shape, "min", inp.min(), "max", inp.max(),
          "mean", inp.mean())

    t = torch.from_numpy(inp)
    with torch.inference_mode():
        out = model(t)  # return_all_feats -> dict with neck_out/backbone_out/head_out(maps)

    # base_model returns the head dict {'maps':...} at inference unless return_all_feats
    # captures more. With return_all_feats=True and eval + dict head, it returns x (the head dict).
    # To get intermediates, run submodules manually:
    with torch.inference_mode():
        feats = model.backbone(t)
        neck = model.neck(feats)
        head = model.head(neck)

    # backbone returns a list of feature maps
    if isinstance(feats, (list, tuple)):
        for i, f in enumerate(feats):
            fn = os.path.join(OUT, f"backbone_{i}.npy")
            np.save(fn, f.detach().cpu().numpy())
            print(f"[backbone_{i}] shape {tuple(f.shape)}")
    else:
        np.save(os.path.join(OUT, "backbone_0.npy"), feats.detach().cpu().numpy())
        print("[backbone] shape", tuple(feats.shape))

    np.save(os.path.join(OUT, "neck.npy"), neck.detach().cpu().numpy())
    print("[neck] shape", tuple(neck.shape))

    maps = head["maps"] if isinstance(head, dict) else head
    maps_np = maps.detach().cpu().numpy()
    np.save(os.path.join(OUT, "maps.npy"), maps_np)
    print("[maps] shape", maps_np.shape, "min", maps_np.min(), "max",
          maps_np.max(), "mean", maps_np.mean())

    # Also write a flat little-endian f32 binary + a header for Rust to read.
    def dump_bin(name, arr):
        arr = np.ascontiguousarray(arr, dtype="<f4")
        with open(os.path.join(OUT, name + ".bin"), "wb") as f:
            f.write(arr.tobytes())
        with open(os.path.join(OUT, name + ".shape"), "w") as f:
            f.write(",".join(str(d) for d in arr.shape))
    dump_bin("input", inp)
    dump_bin("neck", neck.detach().cpu().numpy())
    dump_bin("maps", maps_np)
    if isinstance(feats, (list, tuple)):
        for i, f in enumerate(feats):
            dump_bin(f"backbone_{i}", f.detach().cpu().numpy())
    print("[done] dumped .npy + .bin to", OUT)

if __name__ == "__main__":
    main()
