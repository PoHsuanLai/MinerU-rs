#!/usr/bin/env python3
"""ONNX ground-truth dumper for SLANet-plus, for the Rust parity test.

Runs the real `slanet-plus.onnx` (via onnxruntime) on the SAME deterministic
488x488 synthetic grid the Rust `slanet_real`/`parity` tests build, and dumps the
per-step structure probabilities `[L, 50]` and loc quads `[L, 8]` as little-endian
f32 `.bin` + `.shape` files. The Rust parity test diffs its hand-ported decoder
against these.

The point of THIS target: the model's autoregressive decode is an ONNX `Loop`
with ~30 carried variables (a Paddle export). The hand-port must reproduce the
Loop's per-step computation EXACTLY — a shape-correct forward that diverges after
a few steps (as an earlier attempt did, matching steps 0-3 then diverging at 4) is
NOT correct. The argmax token sequence on this grid is, per ort:

    [5, 48, 48, 48, 48, 6] * 4, then 49 (eos), 0   (26 steps)
    == (<tr> <td></td> <td></td> <td></td> <td></td> </tr>) * 4, eos

onnxruntime + onnx are not in the system Python (PEP-668); this was run in a venv
with `pip install onnx onnxruntime numpy`. Requires SLANET_ONNX or the default
on-disk path.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

ONNX = os.environ.get(
    "SLANET_ONNX",
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/SlanetPlus/slanet-plus.onnx",
)
OUT = os.path.dirname(os.path.abspath(__file__))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def synthetic_grid(h=488, w=488):
    """The same deterministic 4x4 grid the Rust tests build."""
    img = np.full((h, w, 3), 255, np.uint8)
    for y in range(0, h, h // 4):
        img[min(y, h - 1), :, :] = 0
    for x in range(0, w, w // 4):
        img[:, min(x, w - 1), :] = 0
    return img


def preprocess(img):
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x, dtype=np.float32)


def dump(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    arr.tofile(os.path.join(OUT, name + ".bin"))
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in arr.shape))


def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = preprocess(synthetic_grid())
    outs = sess.run(None, {name: x})
    by = {o.name: np.asarray(v) for o, v in zip(sess.get_outputs(), outs)}
    struct = next(v for v in by.values() if v.ndim == 3 and v.shape[-1] == 50)[0]
    loc = next(v for v in by.values() if v.ndim == 3 and v.shape[-1] == 8)[0]
    dump("slanet_structure", struct)
    dump("slanet_loc", loc)
    print("structure", struct.shape, "loc", loc.shape)
    print("argmax seq:", struct.argmax(-1).tolist())


if __name__ == "__main__":
    main()
