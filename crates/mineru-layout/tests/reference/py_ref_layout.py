#!/usr/bin/env python3
"""Python reference forward for PP-DocLayoutV2, for Rust `mineru-layout` parity.

Builds the HuggingFace PyTorch reference (`PPDocLayoutV2ForObjectDetection`, from
`mineru/model/layout/pp_doclayoutv2.py`), loads the SAME safetensors checkpoint the
Rust crate loads, runs a DETERMINISTIC synthetic 800x800 RGB image through the exact
`/255` rescale the Rust `preprocess` uses (no mean/std), forwards it, and dumps the
per-stage activations as little-endian f32 `<name>.bin` + `<name>.shape` files:

  - input                          [1, 3, 800, 800]   (the preprocessed input tensor)
  - backbone_0 / _1 / _2           HGNetV2 stage2/3/4 feature maps, ch [512,1024,2048]
  - proj_0 / _1 / _2               encoder_input_proj outputs (1x1 conv + bn), 256-ch
  - encoder_0 / _1 / _2            hybrid-encoder (AIFI + CCFM) fused maps, 256-ch
  - logits                         [1, 300, 25]  final decoder class logits
  - pred_boxes                     [1, 300, 4]   final decoder boxes (cxcywh in [0,1])
  - order_logits                   [1, 300, 300] reading-order pairwise logits

The input is 800x800 = the model input size, so the torchvision BICUBIC resize is an
identity resize (source already at target). This isolates conv / BN / attention math
from resize-interpolation differences, which is the point of THIS parity target. The
Rust parity test builds the same deterministic input tensor directly (same `/255`
rescale) so preprocessing is not a variable in the model-math comparison.

Env:
  LAYOUT_WEIGHTS   directory containing model.safetensors + config.json
                   (default: the on-disk PEK-1.0 location).

Requires: torch, torchvision, transformers>=4.57.3,<5, safetensors, numpy. A stub is
installed for `mineru.utils.bbox_utils` (used only by postprocessing, not the forward)
so the reference module imports without the full `mineru` package.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

# ---- Checkpoint dir: override with LAYOUT_WEIGHTS ----------------------------------
CKPT_DIR = os.environ.get(
    "LAYOUT_WEIGHTS",
    "/Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/Layout/PP-DocLayoutV2",
)
# Reference tensors are written next to this script; point the Rust parity test's
# LAYOUT_REF_DIR at this directory.
OUT = os.path.dirname(os.path.abspath(__file__))

# Path to the committed HF reference implementation.
PP_REF = "/Users/pohsuanlai/Documents/mineru/mineru/mineru/model/layout/pp_doclayoutv2.py"

INPUT_SIZE = 800


def _install_stubs():
    """Stub `mineru.utils.bbox_utils` so the reference imports without the full pkg.

    `normalize_to_int_bbox` is only used in the layout-model postprocessing (box
    clipping), never in the network forward pass we are validating.
    """
    m_mineru = types.ModuleType("mineru")
    m_utils = types.ModuleType("mineru.utils")
    m_bbox = types.ModuleType("mineru.utils.bbox_utils")
    m_bbox.normalize_to_int_bbox = lambda box, image_size=None: [int(v) for v in box]
    sys.modules.setdefault("mineru", m_mineru)
    sys.modules.setdefault("mineru.utils", m_utils)
    sys.modules["mineru.utils.bbox_utils"] = m_bbox


def load_reference_module():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("pp_doclayoutv2", PP_REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(mod):
    cfg = mod.PPDocLayoutV2Config.from_pretrained(CKPT_DIR)
    model = mod.PPDocLayoutV2ForObjectDetection.from_pretrained(CKPT_DIR, config=cfg)
    model.eval()
    return model


# ---- deterministic synthetic image: 800x800 RGB gradient/pattern ------------------
def make_image(h=INPUT_SIZE, w=INPUT_SIZE):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = (xx * 255 // (w - 1)).astype(np.uint8)
    g = (yy * 255 // (h - 1)).astype(np.uint8)
    b = (((xx + yy) * 255 // (h + w - 2))).astype(np.uint8)
    img = np.stack([r, g, b], axis=2)  # HWC, RGB, uint8
    return img


def preprocess(img_rgb_hwc):
    """Match Rust `mineru_layout::preprocess`: RGB, CHW, plain `* 1/255`, NO mean/std.

    The input is already 800x800 so the resize is an identity; we skip torchvision and
    build the tensor directly (bit-for-bit what the Rust test builds).
    """
    x = img_rgb_hwc.astype(np.float32) / 255.0  # HWC, RGB
    x = np.transpose(x, (2, 0, 1))  # CHW (RGB)
    x = x[None, ...]  # NCHW
    return np.ascontiguousarray(x, dtype=np.float32)


def dump_bin(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(os.path.join(OUT, name + ".bin"), "wb") as f:
        f.write(arr.tobytes())
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in arr.shape))
    print(f"[{name}] shape {tuple(arr.shape)}  "
          f"min {arr.min():.4e} max {arr.max():.4e} mean {arr.mean():.4e}")


def main():
    mod = load_reference_module()
    model = build_model(mod)
    print("[build] model built + checkpoint loaded")

    img = make_image()
    inp = preprocess(img)

    caps = {}

    def mkhook(name):
        def h(_m, _i, o):
            caps[name] = o
        return h

    # Backbone: RTDetrConvEncoder returns a list of (feature_map, mask) per level.
    model.model.backbone.register_forward_hook(mkhook("backbone"))
    # encoder_input_proj[i]: 1x1 conv + bn -> 256-ch.
    for i in range(3):
        model.model.encoder_input_proj[i].register_forward_hook(mkhook(f"proj_{i}"))
    # Hybrid encoder: returns BaseModelOutput whose last_hidden_state is a list of 3 maps.
    model.model.encoder.register_forward_hook(mkhook("encoder"))

    t = torch.from_numpy(inp)
    with torch.inference_mode():
        out = model(t)

    dump_bin("input", inp)

    backbone = caps["backbone"]
    for i, entry in enumerate(backbone):
        feat = entry[0] if isinstance(entry, (list, tuple)) else entry
        dump_bin(f"backbone_{i}", feat.detach().cpu().numpy())

    for i in range(3):
        dump_bin(f"proj_{i}", caps[f"proj_{i}"].detach().cpu().numpy())

    enc = caps["encoder"]
    enc_maps = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc
    for i, f in enumerate(enc_maps):
        dump_bin(f"encoder_{i}", f.detach().cpu().numpy())

    # ---- Detector tail + reading-order, recomputed under a STABLE sort -----------
    # `PPDocLayoutV2ForObjectDetection.forward` reorders the raw decoder outputs by a
    # threshold keep-mask sort so kept queries float to the front, and returns
    # `logits`, `pred_boxes`, and `order_logits` all on that SAME query axis (the
    # postprocessor runs a single topk over the flattened logits and gathers boxes
    # and order sequences by that one index, so it is invariant to the permutation).
    #
    # The wrapper uses `torch.argsort(descending=True)`, which is NON-stable: its
    # tie-break among the equal keep-flags is implementation-defined and not
    # reproducible across backends. So we reproduce the exact same tail here but with
    # `stable=True` — kept queries first in original order, then dropped in original
    # order. This is the reproducible canonical form the Rust `sort_and_read_order`
    # matches. All three dumped tensors stay mutually consistent because they are
    # produced by one shared permutation, exactly as in the reference forward; only
    # the (semantically irrelevant) tie-break differs from the wrapper's own output.
    raw_boxes = out.intermediate_reference_points[:, -1]   # [1, 300, 4]
    raw_logits = out.intermediate_logits[:, -1]            # [1, 300, 25]

    box_centers, box_sizes = raw_boxes.split(2, dim=-1)
    bboxes = torch.cat([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], dim=-1) * 1000.0
    bboxes = bboxes.clamp(0.0, 1000.0)

    max_logits, class_ids = raw_logits.max(dim=-1)
    mask = max_logits.sigmoid() >= model._class_thresholds_tensor[class_ids]
    indices = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)

    def _gather(t, feat):
        return torch.take_along_dim(t, indices[..., None].expand(-1, -1, feat), dim=1)

    sorted_logits = _gather(raw_logits, raw_logits.size(-1))
    sorted_pred_boxes = _gather(raw_boxes, 4)
    sorted_boxes = _gather(bboxes, 4)
    sorted_class_ids = torch.take_along_dim(class_ids, indices, dim=1)
    sorted_mask = torch.take_along_dim(mask, indices, dim=1)

    pad_boxes = torch.where(sorted_mask[..., None], sorted_boxes, torch.zeros_like(sorted_boxes))
    pad_class_ids = model._class_order_tensor[
        torch.where(sorted_mask, sorted_class_ids, torch.zeros_like(sorted_class_ids))
    ]
    order_logits = model.reading_order(boxes=pad_boxes, labels=pad_class_ids, mask=mask)[
        :, :, : model.num_queries
    ]

    dump_bin("logits", sorted_logits.detach().cpu().numpy())
    dump_bin("pred_boxes", sorted_pred_boxes.detach().cpu().numpy())
    dump_bin("order_logits", order_logits.detach().cpu().numpy())

    print("[done] dumped .bin/.shape to", OUT)


if __name__ == "__main__":
    main()
