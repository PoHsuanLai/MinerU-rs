#!/usr/bin/env python3
"""Convert the SLANet-plus ONNX graph's constant weights to the flat
`.safetensors` the Rust `mineru-table::slanet` model loads at runtime.

The SLANet-plus ONNX cannot be `burn-onnx`-codegen'd (its autoregressive decoder
is an ONNX `Loop`, and burn-onnx 0.21's type inference fails on the surrounding
`ConstantOfShape` nodes), so the whole network is hand-ported in Burn and its
weights are supplied as a flat safetensors whose keys match the Burn module field
paths. This script produces that file:

    python convert_slanet_weights.py \
        /Volumes/Archive/mineru/models/PDF-Extract-Kit-1.0/models/TabRec/SlanetPlus/slanet-plus.onnx

It writes `slanet-plus.safetensors` next to the input `.onnx`, which is exactly
where `SlaNet::load(<onnx path>)` looks for it.

Key contract (Burn field path -> tensor):
  * `conv.<N>.weight` / `conv.<N>.bias`  — backbone/neck conv `Conv.N`
      (only the four squeeze-excite 1x1 convs 24/25/28/29 carry a bias)
  * `bn.<M>.{weight,bias,running_mean,running_var}` — batch-norm `BatchNormalization.M`
  * `head.linear{0..6}.weight` / `.bias`  — SLAHead linears, TRANSPOSED from the
      Paddle `[in, out]` layout to PyTorch/PtLinear `[out, in]`
      (linear0/linear2 have no bias)
  * `head.gru.{w_ih,w_hh,b_ih,b_hh}`      — GRU cell, `[3*H, in]` / `[3*H, H]`

The `conv.*` / `bn.*` keys are re-prefixed to `backbone.*` by the Rust loader's
`KeyRemap` before matching the module tree.

Requires: onnx, numpy, safetensors.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def main(onnx_path: str) -> None:
    model = onnx.load(onnx_path)
    graph = model.graph

    # All Constant tensor values in the main graph (backbone/neck weights).
    const = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const[node.output[0]] = numpy_helper.to_array(attr.t)

    out: dict[str, np.ndarray] = {}

    # Convs + batch-norms, keyed by ONNX node index.
    for node in graph.node:
        if node.op_type == "Conv":
            idx = int(node.name.split(".")[1])
            out[f"conv.{idx}.weight"] = const[node.input[1]].astype(np.float32)
            if len(node.input) >= 3:  # SE 1x1 convs carry a bias
                out[f"conv.{idx}.bias"] = const[node.input[2]].astype(np.float32)
        elif node.op_type == "BatchNormalization":
            idx = int(node.name.split(".")[1])
            out[f"bn.{idx}.weight"] = const[node.input[1]].astype(np.float32)
            out[f"bn.{idx}.bias"] = const[node.input[2]].astype(np.float32)
            out[f"bn.{idx}.running_mean"] = const[node.input[3]].astype(np.float32)
            out[f"bn.{idx}.running_var"] = const[node.input[4]].astype(np.float32)

    # SLAHead weights are main-graph constants fed into the ONNX Loop as inputs.
    def w(name: str) -> np.ndarray:
        return const[name].astype(np.float32)

    # Paddle linear weight is [in, out]; PtLinear stores [out, in].
    out["head.linear0.weight"] = w("linear_0.w_0").T.copy()  # no bias
    out["head.linear1.weight"] = w("linear_1.w_0").T.copy()
    out["head.linear1.bias"] = w("linear_1.b_0")
    out["head.linear2.weight"] = w("linear_2.w_0").T.copy()  # no bias
    out["head.linear3.weight"] = w("linear_3.w_0").T.copy()
    out["head.linear3.bias"] = w("linear_3.b_0")
    out["head.linear4.weight"] = w("linear_4.w_0").T.copy()
    out["head.linear4.bias"] = w("linear_4.b_0")
    out["head.linear5.weight"] = w("linear_5.w_0").T.copy()
    out["head.linear5.bias"] = w("linear_5.b_0")
    out["head.linear6.weight"] = w("linear_6.w_0").T.copy()
    out["head.linear6.bias"] = w("linear_6.b_0")

    # GRU cell keeps the Paddle stacked [3*H, in] / [3*H, H] layout.
    out["head.gru.w_ih"] = w("gru_cell_0.w_0").copy()
    out["head.gru.w_hh"] = w("gru_cell_0.w_1").copy()
    out["head.gru.b_ih"] = w("gru_cell_0.b_0")
    out["head.gru.b_hh"] = w("gru_cell_0.b_1")

    dst = Path(onnx_path).with_suffix(".safetensors")
    save_file(out, str(dst))
    print(f"wrote {len(out)} tensors -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
