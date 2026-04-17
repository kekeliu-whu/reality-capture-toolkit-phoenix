"""
Export ALIKED model to ONNX format for TensorRT deployment.

Exports two ONNX models:
  1. aliked_backbone.onnx  — image → (feature_map, score_map)
  2. aliked_sddh.onnx      — (feature_map, keypoints_wh) → descriptors

Usage:
    cd ALIKED
    python ../feature_extraction/tools/export_aliked_onnx.py \
        --model aliked-n16rot \
        --output ../feature_extraction/models/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper

# Add ALIKED directory to path
ALIKED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ALIKED')
sys.path.insert(0, os.path.abspath(ALIKED_DIR))

from nets.aliked import ALIKED
from nets.blocks import DeformableConv2d


# ---------------------------------------------------------------------------
# Register ONNX symbolic for torchvision::deform_conv2d  (maps to onnx::DeformConv, opset 19+)
# ---------------------------------------------------------------------------
def _deform_conv2d_symbolic(g, input, weight, offset, mask, bias,
                            stride_h, stride_w,
                            pad_h, pad_w,
                            dil_h, dil_w,
                            n_weight_grps, n_offset_grps,
                            use_mask):
    """Map torchvision::deform_conv2d to ONNX DeformConv operator (opset 19)."""
    stride_h = symbolic_helper._parse_arg(stride_h, "i")
    stride_w = symbolic_helper._parse_arg(stride_w, "i")
    pad_h = symbolic_helper._parse_arg(pad_h, "i")
    pad_w = symbolic_helper._parse_arg(pad_w, "i")
    dil_h = symbolic_helper._parse_arg(dil_h, "i")
    dil_w = symbolic_helper._parse_arg(dil_w, "i")
    n_weight_grps = symbolic_helper._parse_arg(n_weight_grps, "i")
    n_offset_grps = symbolic_helper._parse_arg(n_offset_grps, "i")

    # ONNX DeformConv spec input order: X, W, offset, [B], [mask]
    return g.op(
        "DeformConv",
        input,
        weight,
        offset,
        bias,
        dilations_i=[dil_h, dil_w],
        group_i=n_weight_grps,
        kernel_shape_i=weight.type().sizes()[2:],
        offset_group_i=n_offset_grps,
        pads_i=[pad_h, pad_w, pad_h, pad_w],
        strides_i=[stride_h, stride_w],
    )

register_custom_op_symbolic("torchvision::deform_conv2d", _deform_conv2d_symbolic, 19)


# Monkey-patch DeformableConv2d.forward to use Python float for max_offset,
# avoiding CPU tensor in clamp() during JIT tracing.
_orig_deform_forward = DeformableConv2d.forward

def _patched_deform_forward(self, x):
    h, w = x.shape[2], x.shape[3]
    max_offset = float(max(int(h), int(w))) / 4.0

    out = self.offset_conv(x)
    if self.mask:
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
    else:
        offset = out
        mask = None
    offset = offset.clamp(-max_offset, max_offset)
    x = torchvision.ops.deform_conv2d(input=x,
                                      offset=offset,
                                      weight=self.regular_conv.weight,
                                      bias=self.regular_conv.bias,
                                      padding=self.padding,
                                      mask=mask)
    return x

DeformableConv2d.forward = _patched_deform_forward


# ---------------------------------------------------------------------------
# Exportable backbone: image → (feature_map, score_map)
# Assumes input image is already padded so H,W are divisible by 32.
# ---------------------------------------------------------------------------
class ALIKEDBackbone(nn.Module):
    def __init__(self, model: ALIKED):
        super().__init__()
        self.pool2 = model.pool2
        self.pool4 = model.pool4
        self.block1 = model.block1
        self.block2 = model.block2
        self.block3 = model.block3
        self.block4 = model.block4
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.conv4 = model.conv4
        self.upsample2 = model.upsample2
        self.upsample8 = model.upsample8
        self.upsample32 = model.upsample32
        self.score_head = model.score_head
        self.gate = model.gate

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [1, 3, H, W]  float32, RGB normalised to [0,1].
                   H, W must be divisible by 32.
        Returns:
            feature_map: [1, C, H, W]  L2-normalised dense feature map.
            score_map:   [1, 1, H, W]  keypoint score map in (0,1).
        """
        # ---- encoder ----
        x1 = self.block1(image)           # [1, c1, H, W]
        x2 = self.pool2(x1)
        x2 = self.block2(x2)             # [1, c2, H/2, W/2]
        x3 = self.pool4(x2)
        x3 = self.block3(x3)             # [1, c3, H/8, W/8]
        x4 = self.pool4(x3)
        x4 = self.block4(x4)             # [1, c4, H/32, W/32]

        # ---- aggregation (FPN-like) ----
        x1 = self.gate(self.conv1(x1))   # [1, dim/4, H, W]
        x2 = self.gate(self.conv2(x2))   # [1, dim/4, H/2, W/2]
        x3 = self.gate(self.conv3(x3))   # [1, dim/4, H/8, W/8]
        x4 = self.gate(self.conv4(x4))   # [1, dim/4, H/32, W/32]

        x2_up = self.upsample2(x2)       # → H x W
        x3_up = self.upsample8(x3)       # → H x W
        x4_up = self.upsample32(x4)      # → H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)  # [1, dim, H, W]

        # ---- heads ----
        score_map = torch.sigmoid(self.score_head(x1234))      # [1, 1, H, W]
        feature_map = F.normalize(x1234, p=2, dim=1)           # [1, dim, H, W]

        return feature_map, score_map


# ---------------------------------------------------------------------------
# Exportable SDDH descriptor head.
# Replaces custom get_patches CUDA kernel with grid_sample.
# ---------------------------------------------------------------------------
class ALIKEDSddh(nn.Module):
    def __init__(self, model: ALIKED):
        super().__init__()
        sddh = model.desc_head
        self.kernel_size = sddh.kernel_size    # 3
        self.n_pos = sddh.n_pos               # 16
        self.offset_conv = sddh.offset_conv
        self.sf_conv = sddh.sf_conv
        self.agg_weights = sddh.agg_weights    # [n_pos, dim, dim]
        dim = sddh.agg_weights.shape[-1]       # 128

        # Pre-compute KxK patch offsets in (dx, dy) pixel space
        radius = (self.kernel_size - 1) / 2.0  # 1.0 for K=3
        x = torch.arange(self.kernel_size, dtype=torch.float32) - radius
        gy, gx = torch.meshgrid(x, x, indexing='ij')
        patch_offsets = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # [K*K, 2]
        self.register_buffer('patch_offsets', patch_offsets)

        # Pre-flatten agg_weights for matmul (replaces einsum)
        # einsum('ncp,pcd->nd') == reshape(N, P*C) @ reshape(P*C, D)
        self.register_buffer(
            'agg_weights_flat',
            sddh.agg_weights.permute(1, 0, 2).reshape(-1, dim))  # [C*P, D]
        # Note: features after permute(0,2,1) gives [N,P,C], flatten → [N, P*C]
        # agg_weights [P,C,D] → we need [P*C, D] but with P as the outer dim:
        # agg_weights.reshape(P*C, D) already has P as outer dim. ✓
        # But features.permute(0,2,1).reshape(N,-1) gives [N, P*C] with P outer. ✓
        self.register_buffer(
            'agg_w',
            sddh.agg_weights.reshape(-1, dim))  # [P*C, D]

    def forward(self, feature_map: torch.Tensor, keypoints_wh: torch.Tensor,
                feature_map_hw: torch.Tensor):
        """
        Args:
            feature_map:    [1, C, H, W]  from backbone.
            keypoints_wh:   [N, 2]  pixel coordinates (x, y).
            feature_map_hw: [2]  float tensor containing [H, W] of feature_map.
        Returns:
            descriptors:  [N, D]  L2-normalised descriptors.
        """
        N = keypoints_wh.shape[0]
        K = self.kernel_size
        C = feature_map.shape[1]

        # Normalisation constants — derived from explicit input to avoid
        # JIT-tracing baking concrete values into the ONNX graph.
        H_val = feature_map_hw[0]
        W_val = feature_map_hw[1]
        w_norm = W_val - 1.0
        h_norm = H_val - 1.0
        max_offset = torch.max(H_val, W_val) / 4.0

        # --- 1. Extract KxK patches via grid_sample ---
        # patch_coords: [N, K*K, 2] in pixel coordinates
        patch_coords = keypoints_wh.unsqueeze(1) + self.patch_offsets.unsqueeze(0)
        # Normalise to [-1, 1]
        patch_grid_x = 2.0 * patch_coords[:, :, 0:1] / w_norm - 1.0
        patch_grid_y = 2.0 * patch_coords[:, :, 1:2] / h_norm - 1.0
        patch_grid = torch.cat([patch_grid_x, patch_grid_y], dim=-1)  # [N, K*K, 2]
        patch_grid = patch_grid.reshape(1, N * K * K, 1, 2)

        sampled = F.grid_sample(feature_map, patch_grid,
                                mode='bilinear', align_corners=True)  # [1, C, N*K*K, 1]
        patches = sampled.reshape(C, N, K, K).permute(1, 0, 2, 3)     # [N, C, K, K]

        # --- 2. Offset estimation ---
        offset = self.offset_conv(patches)               # [N, 2*n_pos, 1, 1]
        offset = offset.clamp(-max_offset, max_offset)
        offset = offset[:, :, 0, 0]                      # [N, 2*n_pos]
        offset = offset.reshape(N, 2, self.n_pos).permute(0, 2, 1)  # [N, n_pos, 2]

        # --- 3. Sample at deformable positions ---
        pos = keypoints_wh.unsqueeze(1) + offset          # [N, n_pos, 2]
        pos_grid_x = 2.0 * pos[:, :, 0:1] / w_norm - 1.0
        pos_grid_y = 2.0 * pos[:, :, 1:2] / h_norm - 1.0
        pos_grid = torch.cat([pos_grid_x, pos_grid_y], dim=-1)  # [N, n_pos, 2]
        pos_grid = pos_grid.reshape(1, N * self.n_pos, 1, 2)

        features = F.grid_sample(feature_map, pos_grid,
                                 mode='bilinear', align_corners=True)  # [1, C, N*n_pos, 1]
        features = features.reshape(C, N, self.n_pos, 1)
        features = features.permute(1, 0, 2, 3)           # [N, C, n_pos, 1]

        # --- 4. SF conv + SELU ---
        features = F.selu(self.sf_conv(features))          # [N, C, n_pos, 1]
        features = features.squeeze(-1)                    # [N, C, n_pos]

        # --- 5. Aggregate (replaces einsum with matmul) ---
        features_flat = features.permute(0, 2, 1).reshape(N, -1)  # [N, P*C]
        descs = features_flat @ self.agg_w                         # [N, D]

        # --- 6. L2-normalise ---
        descs = F.normalize(descs, p=2.0, dim=1)

        return descs


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_backbone(model_orig: ALIKED, backbone: ALIKEDBackbone, device: str):
    """Compare backbone output with original model's extract_dense_map."""
    H, W = 480, 640
    img = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        fm_orig, sm_orig = model_orig.extract_dense_map(img)
        fm_new, sm_new = backbone(img)
    fm_err = (fm_orig - fm_new).abs().max().item()
    sm_err = (sm_orig - sm_new).abs().max().item()
    print(f"  Backbone validation: feature_map max_err={fm_err:.2e}, score_map max_err={sm_err:.2e}")
    assert fm_err < 1e-5 and sm_err < 1e-5, "Backbone output mismatch!"


def validate_sddh(model_orig: ALIKED, sddh_export: ALIKEDSddh, device: str):
    """Compare SDDH output with original model on random keypoints."""
    H, W = 480, 640
    img = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        fm, sm = model_orig.extract_dense_map(img)
        # Generate some random keypoints in [-1, 1] normalised coords
        N = 200
        kpts_norm = (torch.rand(N, 2, device=device) * 2 - 1) * 0.9  # avoid borders
        # Original SDDH
        descs_orig, _ = model_orig.desc_head(fm, [kpts_norm])
        descs_orig = descs_orig[0]  # [N, D]
        # Convert to pixel coords for export model
        wh = torch.tensor([[W - 1.0, H - 1.0]], device=device)
        kpts_wh = (kpts_norm / 2 + 0.5) * wh
        fm_hw = torch.tensor([float(H), float(W)], dtype=torch.float32, device=device)
        descs_new = sddh_export(fm, kpts_wh, fm_hw)

    cos_sim = (descs_orig * descs_new).sum(dim=1)
    print(f"  SDDH validation: cosine similarity mean={cos_sim.mean():.4f}, min={cos_sim.min():.4f}")
    assert cos_sim.mean() > 0.95, "SDDH output mismatch!"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_backbone(model: ALIKED, output_dir: str, device: str, opset: int = 16):
    print("Exporting ALIKED backbone...")
    backbone = ALIKEDBackbone(model).to(device).eval()
    validate_backbone(model, backbone, device)

    H, W = 480, 640  # reference size; dynamic axes allow other sizes
    dummy = torch.randn(1, 3, H, W, device=device)
    path = os.path.join(output_dir, 'aliked_backbone.onnx')

    torch.onnx.export(
        backbone,
        (dummy,),
        path,
        opset_version=opset,
        input_names=['image'],
        output_names=['feature_map', 'score_map'],
        dynamic_axes={
            'image':       {2: 'H', 3: 'W'},
            'feature_map': {2: 'H', 3: 'W'},
            'score_map':   {2: 'H', 3: 'W'},
        },
        dynamo=False,
    )
    print(f"  Saved: {path}")
    return path


def export_sddh(model: ALIKED, output_dir: str, device: str, opset: int = 16):
    print("Exporting ALIKED SDDH descriptor head...")
    sddh = ALIKEDSddh(model).to(device).eval()
    validate_sddh(model, sddh, device)

    C = model.desc_head.agg_weights.shape[-1]  # 128
    H, W, N = 480, 640, 500
    dummy_fm = torch.randn(1, C, H, W, device=device)
    dummy_kpts = torch.rand(N, 2, device=device) * torch.tensor([[W - 1.0, H - 1.0]], device=device)
    dummy_hw = torch.tensor([float(H), float(W)], dtype=torch.float32, device=device)
    path = os.path.join(output_dir, 'aliked_sddh.onnx')

    torch.onnx.export(
        sddh,
        (dummy_fm, dummy_kpts, dummy_hw),
        path,
        opset_version=opset,
        input_names=['feature_map', 'keypoints_wh', 'feature_map_hw'],
        output_names=['descriptors'],
        dynamic_axes={
            'feature_map':  {2: 'H', 3: 'W'},
            'keypoints_wh': {0: 'N'},
            'descriptors':  {0: 'N'},
        },
        dynamo=False,
    )
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Export ALIKED to ONNX')
    parser.add_argument('--model', default='aliked-n16rot',
                        choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='../feature_extraction/models/',
                        help='Output directory for ONNX models')
    parser.add_argument('--opset', type=int, default=19)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading ALIKED model: {args.model}")
    model = ALIKED(model_name=args.model, device=args.device, top_k=5000, scores_th=0.2, n_limit=5000)
    model.eval()

    export_backbone(model, args.output, args.device, args.opset)
    export_sddh(model, args.output, args.device, args.opset)
    print("\nDone! Next step: convert ONNX models to TensorRT engines.")


if __name__ == '__main__':
    main()
