"""
Export LightGlue matcher to ONNX format for TensorRT deployment.

LightGlue requires modifications for static-graph export:
  - Disable FlashAttention (use standard scaled-dot-product attention)
  - Disable adaptive early stopping (run all Transformer layers)
  - Flatten dict-based I/O to plain tensors

Usage:
    python sfm-phoenix/tools/export_lightglue_onnx.py \
        --output sfm-phoenix/models/
"""

import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import TensorProto, helper


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

try:
    from lightglue.lightglue import LightGlue
except ImportError:
    print("ERROR: lightglue not installed.  pip install lightglue")
    sys.exit(1)


class LightGlueExportable(nn.Module):
    """
    Thin wrapper around LightGlue that:
      1. Takes flat tensor inputs instead of nested dicts.
      2. Disables flash attention & early stopping for ONNX compatibility.
      3. Returns match indices and scores as tensors.
    """

    def __init__(self, features: str = 'aliked', device: str = 'cuda'):
        super().__init__()
        self.lg = LightGlue(features=features).to(device).eval()
        # Disable features incompatible with static graph export
        self.lg.conf.flash = False
        if hasattr(self.lg.conf, 'depth_confidence'):
            self.lg.conf.depth_confidence = -1  # run all layers
        if hasattr(self.lg.conf, 'width_confidence'):
            self.lg.conf.width_confidence = -1
        self.eval()

    def forward(
        self,
        kpts0: torch.Tensor,
        desc0: torch.Tensor,
        kpts1: torch.Tensor,
        desc1: torch.Tensor,
    ):
        """
        Args:
            kpts0: [1, N, 2]  keypoints of image 0 in pixel coords
            desc0: [1, N, D]  descriptors of image 0
            kpts1: [1, M, 2]  keypoints of image 1 in pixel coords
            desc1: [1, M, D]  descriptors of image 1
        Returns:
            matches0: [1, N]  index into kpts1 for each kpt in image 0 (-1 = unmatched)
            mscores0: [1, N]  matching confidence scores
        """
        data = {
            'image0': {'keypoints': kpts0, 'descriptors': desc0},
            'image1': {'keypoints': kpts1, 'descriptors': desc1},
        }
        with torch.no_grad():
            result = self.lg(data)
        return result['matches0'], result['matching_scores0']


def validate(wrapper: LightGlueExportable, device: str):
    """Smoke test with B=1 and B=2 to verify dynamic batch works."""
    for B, N, M in [(1, 200, 300), (2, 150, 250)]:
        kpts0 = torch.rand(B, N, 2, device=device) * 640
        desc0 = F.normalize(torch.randn(B, N, 128, device=device), dim=-1)
        kpts1 = torch.rand(B, M, 2, device=device) * 640
        desc1 = F.normalize(torch.randn(B, M, 128, device=device), dim=-1)
        matches0, mscores0 = wrapper(kpts0, desc0, kpts1, desc1)
        n_matches = (matches0 >= 0).sum().item()
        print(f"  Validation B={B}: {n_matches} matches from {B}x{N} kpts")
        assert matches0.shape == (B, N), f"Expected ({B}, {N}), got {matches0.shape}"
        assert mscores0.shape == (B, N), f"Expected ({B}, {N}), got {mscores0.shape}"


def _make_batch_dynamic(model: onnx.ModelProto, dim_param: str = "B") -> int:
    """Change input/output dim[0] from static B_dummy to dynamic dim_param.

    Returns the number of tensors patched.
    """
    io_names = {"kpts0", "desc0", "kpts1", "desc1", "matches0", "mscores0"}
    changed = 0
    for t in list(model.graph.input) + list(model.graph.output):
        if t.name in io_names:
            d = t.type.tensor_type.shape.dim[0]
            if d.HasField("dim_value"):  # currently a concrete integer
                d.ClearField("dim_value")
                d.dim_param = dim_param
                changed += 1
    return changed


def _fix_reshape_batch_nodes(
    model: onnx.ModelProto, batch_value: int = 2
) -> int:
    """Fix Reshape/Expand nodes whose shape Concat starts with a hardcoded batch.

    When torch.export uses B_dummy=2, some Concat nodes that build Reshape
    shape tensors embed a static [2] constant as their first element, e.g.:
      Concat([2], [N, 4, 64, 3]) -> shape -> Reshape(data, shape)
    TRT rejects these at B_profile_min != batch_value (wrong volume) and at
    B > batch_value the Reshape silently produces the wrong tensor layout.

    Fix: for each such Concat, replace the hardcoded batch constant with a
    dynamic Shape(kpts0, start=0, end=1) result (a 1-element int64 tensor).

    Returns the number of Concat nodes patched.
    """
    graph = model.graph
    first_input_name = graph.input[0].name  # kpts0

    # Collect 1-element int64 initializers whose value == batch_value
    batch_const_names: set = set()
    for init in graph.initializer:
        if init.data_type in (TensorProto.INT64, TensorProto.INT32):
            arr = nph.to_array(init)
            if arr.shape == (1,) and int(arr[0]) == batch_value:
                batch_const_names.add(init.name)

    if not batch_const_names:
        return 0

    # Build output_name -> node lookup
    output_to_node: dict = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Find Reshape/Expand nodes whose shape comes from Concat(batch_const, ...)
    to_fix: list = []  # [(consumer_node, concat_node, batch_const_name)]
    for node in graph.node:
        if node.op_type not in ("Reshape", "Expand"):
            continue
        if len(node.input) < 2:
            continue
        shape_inp = node.input[1]
        if shape_inp not in output_to_node:
            continue
        producer = output_to_node[shape_inp]
        if producer.op_type != "Concat":
            continue
        if not producer.input or producer.input[0] not in batch_const_names:
            continue
        to_fix.append((node, producer, producer.input[0]))

    if not to_fix:
        return 0

    # Insert Shape(kpts0, start=0, end=1) -> 1-element int64 tensor [B]
    dyn_batch_name = "__dyn_batch_1d__"
    shape_node = helper.make_node(
        "Shape",
        inputs=[first_input_name],
        outputs=[dyn_batch_name],
        name="__shape_batch_dim__",
        start=0,
        end=1,
    )
    graph.node.insert(0, shape_node)

    # Patch each unique Concat node: replace its first input with [B]
    patched: set = set()
    for (_consumer, concat_node, _batch_const) in to_fix:
        if concat_node.name in patched:
            continue
        del concat_node.input[0]
        concat_node.input.insert(0, dyn_batch_name)
        patched.add(concat_node.name)

    return len(patched)


def export(wrapper: LightGlueExportable, output_dir: str, device: str):
    print("Exporting LightGlue...")
    validate(wrapper, device)

    # Use B=2 dummy so torch.export does not specialize the batch dim to
    # the static value 1 (which happens when min==input_value==1).
    B_dummy, N, M, D = 2, 500, 500, 128
    kpts0 = torch.rand(B_dummy, N, 2, device=device) * 640
    desc0 = F.normalize(torch.randn(B_dummy, N, D, device=device), dim=-1)
    kpts1 = torch.rand(B_dummy, M, 2, device=device) * 640
    desc1 = F.normalize(torch.randn(B_dummy, M, D, device=device), dim=-1)

    path = os.path.join(output_dir, 'lightglue.onnx')
    external_data_path = path + '.data'

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Use torch.export.Dim to declare dynamic axes for the dynamo exporter.
    # B: batch size. Phoenix runtime supports {1, 4, 8} and caps the engine
    # profile at 8.
    # N, M: number of keypoints in image 0 and image 1 (vary per image)
    from torch.export import Dim
    dim_B = Dim("B", min=1, max=8)
    dim_N = Dim("N", min=1, max=8192)
    dim_M = Dim("M", min=1, max=8192)
    dynamic_shapes = {
        "kpts0": {0: dim_B, 1: dim_N},
        "desc0": {0: dim_B, 1: dim_N},
        "kpts1": {0: dim_B, 1: dim_M},
        "desc1": {0: dim_B, 1: dim_M},
    }

    try:
        # dynamo=True: use torch.export-based exporter (PyTorch 2.x).
        # 'dynamic_shapes' is required instead of 'dynamic_axes' for proper
        # symbolic shape propagation and correct TensorRT profile generation.
        torch.onnx.export(
            wrapper,
            (kpts0, desc0, kpts1, desc1),
            path,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
            input_names=['kpts0', 'desc0', 'kpts1', 'desc1'],
            output_names=['matches0', 'mscores0'],
        )

        model = onnx.load(path, load_external_data=True)
        onnx.save_model(model, path, save_as_external_data=False)

        if os.path.exists(external_data_path):
            os.remove(external_data_path)

        # Patch 1: fix I/O dim[0] annotations (dynamo specialises to B_dummy)
        n_io = _make_batch_dynamic(model)
        # Patch 2: fix internal Reshape/Expand nodes whose shape Concat starts
        # with a hardcoded batch constant — this is what allows B > B_dummy
        # (e.g. B=32) to work correctly in TensorRT.
        n_reshape = _fix_reshape_batch_nodes(model)
        print(f"  Patched batch dims: {n_io} I/O tensor(s), "
              f"{n_reshape} Reshape/Expand node(s)")

        onnx.checker.check_model(model)
        onnx.save(model, path)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"\n  ONNX export failed: {e}")
        print("  Saving TorchScript as fallback...")

        # Fallback: try TorchScript trace
        try:
            traced = torch.jit.trace(wrapper, (kpts0, desc0, kpts1, desc1))
            ts_path = os.path.join(output_dir, 'lightglue.pt')
            traced.save(ts_path)
            print(f"  TorchScript saved: {ts_path}")
        except Exception as e2:
            print(f"  TorchScript also failed: {e2}")
            print("  Please use LightGlue-ONNX: https://github.com/fabio-sim/LightGlue-ONNX")

    return path


def main():
    parser = argparse.ArgumentParser(description='Export LightGlue to ONNX')
    parser.add_argument('--features', default='aliked', choices=['aliked', 'superpoint', 'disk'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='sfm-phoenix/models/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    wrapper = LightGlueExportable(features=args.features, device=args.device)
    export(wrapper, args.output, args.device)
    print("\nDone!")


if __name__ == '__main__':
    main()
