#!/usr/bin/env python3
"""ONNX ground-truth dumper for PP-LCNet_x1_0 table classifier, for the Rust
parity test.

Runs the real `PP-LCNet_x1_0_table_cls.onnx` (via onnxruntime) on the SAME
deterministic synthetic table image the Rust `lcnet_real` test builds, and dumps
the model's 2-class output vector as a little-endian f32 `.bin` + `.shape`. The
Rust parity test diffs the generated Burn forward against this.

The point of THIS target: prove the burn-onnx-generated PP-LCNet forward
reproduces the real ONNX numerically, not just structurally (the CNN is 32 Convs
+ HardSigmoid + a Softmax head).

IMPORTANT — this is NOT raw logits. The ONNX graph's final op is `Softmax`, so
`fetch_name_0` is a 2-class *probability* vector in [0, 1]. The burn-onnx codegen
imports that Softmax too, so the Rust `model.forward(x)` likewise returns
post-softmax probabilities. We dump exactly what both forwards emit (post-softmax)
and compare those; the file is still named `lcnet_logits` for symmetry with the
other gates. The Rust `cls::head` picks the argmax and reports the winning value
as `score` — consistent with these being probabilities.

The preprocessing here mirrors `cls.rs::preprocess` EXACTLY (not PIL/cv2):
  1. shortest-side resize to 256 via nearest-neighbor with the half-pixel
     center mapping `src = round((dst + 0.5)/scale - 0.5)` (clamped),
  2. center-crop 224x224,
  3. per-channel normalize `px * (scale/std) - mean/std`, scale = 1/255,
     ImageNet mean/std,
  4. HWC -> CHW, NCHW batch of 1.

onnxruntime + onnx are not in the system Python (PEP-668); run in a venv with
`pip install onnx onnxruntime numpy`. Requires LCNET_ONNX or the default on-disk
path.
"""
import os

import numpy as np
import onnxruntime as ort

ONNX = os.environ.get(
    "LCNET_ONNX",
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx",
)
OUT = os.path.dirname(os.path.abspath(__file__))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCALE = np.float32(0.003_921_568_6)  # 1/255

RESIZE_SHORT = 256
CROP = 224


def synthetic_table(w=300, h=300):
    """The same deterministic grid the Rust `lcnet_real` test builds:
    white with a black ruling every quarter across width and height."""
    img = np.full((h, w, 3), 255, np.uint8)
    sh = max(h // 4, 1)
    sw = max(w // 4, 1)
    for y in range(0, h, sh):
        img[min(y, h - 1), :, :] = 0
    for x in range(0, w, sw):
        img[:, min(x, w - 1), :] = 0
    return img


def preprocess(img):
    """Byte-for-byte replica of `cls.rs::preprocess` (nearest-neighbor resize,
    NOT PIL/cv2 bilinear)."""
    h, w = img.shape[:2]
    scale = RESIZE_SHORT / float(min(w, h))
    rw = max(int(round(w * scale)), 1)
    rh = max(int(round(h * scale)), 1)
    if rw < CROP or rh < CROP:
        raise ValueError(f"resized {rw}x{rh} smaller than crop {CROP}")

    x1 = (rw - CROP) // 2
    y1 = (rh - CROP) // 2

    chw = np.zeros((3, CROP, CROP), np.float32)
    for cy in range(CROP):
        ry = y1 + cy
        sy = int(round((ry + 0.5) / scale - 0.5))
        sy = min(max(sy, 0), h - 1)
        for cx in range(CROP):
            rx = x1 + cx
            sx = int(round((rx + 0.5) / scale - 0.5))
            sx = min(max(sx, 0), w - 1)
            px = img[sy, sx].astype(np.float32)
            for c in range(3):
                chw[c, cy, cx] = px[c] * (SCALE / STD[c]) - MEAN[c] / STD[c]
    return np.ascontiguousarray(chw[None], dtype=np.float32)


def dump(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    arr.tofile(os.path.join(OUT, name + ".bin"))
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in arr.shape))


def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = preprocess(synthetic_table())
    outs = sess.run(None, {name: x})
    out = np.asarray(outs[0])[0]  # [2]
    assert out.shape == (2,), f"unexpected output shape {out.shape}"
    dump("lcnet_logits", out)
    print("lcnet output (post-softmax probs):", out.tolist())
    print("argmax:", int(out.argmax()))


if __name__ == "__main__":
    main()
