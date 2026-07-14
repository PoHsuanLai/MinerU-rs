#!/usr/bin/env python3
"""ONNX ground-truth dumper for the UNet line segmenter, for the Rust parity test.

Runs the real `unet.onnx` (via onnxruntime) on the SAME preprocessed 1024x1024
input tensor the Rust `unet_real` test produced, and dumps the resulting per-pixel
class mask as a little-endian **int32** `.bin` + `.shape`. The Rust parity test
diffs the generated Burn forward's mask against this.

The point of THIS target: prove the burn-onnx-generated UNet forward reproduces
the real ONNX segmentation — a deep conv / convtranspose / resize encoder-decoder
(73 Convs, 2 ConvTransposes, 5 Resizes) whose graph ends in
`... ReduceMax, Sub, Exp, ReduceSum, Div, ArgMax, Unsqueeze`. The ONNX output
`output` is therefore the **argmaxed int class mask** of shape `[1, N, H, W]`
(int64), 0 = background / 1 = horizontal line / 2 = vertical line. The generated
Rust `forward` likewise returns the argmaxed int mask (its top-level forward ends
in the same argmax+unsqueeze), so there is NO pre-argmax logit volume exposed on
the Rust side. We therefore dump the argmaxed mask and the Rust gate asserts
per-pixel agreement (target: exact / >=99.9%).

WHY WE CONSUME A DUMPED INPUT rather than preprocess here: the Rust preprocess
resizes with the `image` crate's separable **Triangle** filter, which is not
byte-identical to any stock numpy/PIL/cv2 resize. To feed onnxruntime the
IDENTICAL tensor the Burn forward sees, the Rust `unet_real` test dumps its
preprocessed input to `unet_input.bin`/`.shape` first; this script loads that and
runs ONNX on it. Run order:

    1. cargo test ... --test unet_real ... --ignored   # dumps unet_input.bin,
                                                        # then SKIPs (no mask yet)
    2. python tests/reference/py_ref_unet.py           # reads input, dumps mask
    3. cargo test ... --test unet_real ... --ignored   # now asserts parity

onnxruntime + onnx are not in the system Python (PEP-668); run in a venv with
`pip install onnx onnxruntime numpy`. Requires UNET_ONNX or the default on-disk
path.
"""
import os

import numpy as np
import onnxruntime as ort

ONNX = os.environ.get(
    "UNET_ONNX",
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/UnetStructure/unet.onnx",
)
OUT = os.path.dirname(os.path.abspath(__file__))


def load_bin(name, dtype):
    path = os.path.join(OUT, name + ".bin")
    shape_path = os.path.join(OUT, name + ".shape")
    if not (os.path.exists(path) and os.path.exists(shape_path)):
        raise SystemExit(
            f"missing {name}.bin/.shape — run the Rust `unet_real` test once first "
            f"to dump the preprocessed input (it SKIPs until this mask exists)."
        )
    with open(shape_path) as f:
        shape = tuple(int(d) for d in f.read().strip().split(","))
    arr = np.fromfile(path, dtype=dtype)
    return arr.reshape(shape)


def dump(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<i4")
    arr.tofile(os.path.join(OUT, name + ".bin"))
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in arr.shape))


def main():
    x = load_bin("unet_input", "<f4").astype(np.float32)
    print("input shape", x.shape)

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = sess.run(None, {name: x})
    mask = np.asarray(outs[0]).astype(np.int64)
    print("raw onnx mask shape", mask.shape, "dtype", mask.dtype)

    # ONNX output is [1, N, H, W]; squeeze to [H, W] (N == 1).
    mask = np.squeeze(mask)
    assert mask.ndim == 2, f"expected a 2-D mask after squeeze, got {mask.shape}"
    dump("unet_mask", mask)

    vals, counts = np.unique(mask, return_counts=True)
    print("mask shape", mask.shape, "classes:", dict(zip(vals.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
