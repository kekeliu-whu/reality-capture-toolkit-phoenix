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

try:
    from lightglue import LightGlue
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
    """Smoke test with random data."""
    N, M, D = 200, 300, 128
    kpts0 = torch.rand(1, N, 2, device=device) * 640
    desc0 = F.normalize(torch.randn(1, N, D, device=device), dim=-1)
    kpts1 = torch.rand(1, M, 2, device=device) * 640
    desc1 = F.normalize(torch.randn(1, M, D, device=device), dim=-1)

    matches0, mscores0 = wrapper(kpts0, desc0, kpts1, desc1)
    n_matches = (matches0 >= 0).sum().item()
    print(f"  Validation: {n_matches} matches from {N} kpts (smoke test with random data)")
    assert matches0.shape == (1, N)
    assert mscores0.shape == (1, N)


def export(wrapper: LightGlueExportable, output_dir: str, device: str, opset: int):
    print("Exporting LightGlue...")
    validate(wrapper, device)

    N, M, D = 500, 500, 128
    kpts0 = torch.rand(1, N, 2, device=device) * 640
    desc0 = F.normalize(torch.randn(1, N, D, device=device), dim=-1)
    kpts1 = torch.rand(1, M, 2, device=device) * 640
    desc1 = F.normalize(torch.randn(1, M, D, device=device), dim=-1)

    path = os.path.join(output_dir, 'lightglue.onnx')

    # LightGlue internals may use operations beyond basic opset.
    # We try opset 17+ for better Transformer op support.
    export_opset = max(opset, 17)

    try:
        torch.onnx.export(
            wrapper,
            (kpts0, desc0, kpts1, desc1),
            path,
            opset_version=export_opset,
            input_names=['kpts0', 'desc0', 'kpts1', 'desc1'],
            output_names=['matches0', 'mscores0'],
            dynamic_axes={
                'kpts0':    {1: 'N'},
                'desc0':    {1: 'N'},
                'kpts1':    {1: 'M'},
                'desc1':    {1: 'M'},
                'matches0': {1: 'N'},
                'mscores0': {1: 'N'},
            },
        )
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"\n  ONNX export failed: {e}")
        print("  LightGlue has complex Transformer internals that may need custom patching.")
        print("  Alternatives:")
        print("    1. Use LightGlue-ONNX project: https://github.com/fabio-sim/LightGlue-ONNX")
        print("    2. Use torch2trt for direct TRT conversion")
        print("    3. Use TorchScript as intermediate: torch.jit.trace()")
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
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    wrapper = LightGlueExportable(features=args.features, device=args.device)
    export(wrapper, args.output, args.device, args.opset)
    print("\nDone!")


if __name__ == '__main__':
    main()
