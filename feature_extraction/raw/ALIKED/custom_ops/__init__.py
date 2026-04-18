import math
import torch
from torch import Tensor
import torch.nn.functional as F

from pathlib import Path

file_path = Path(__file__)
_loaded = False
for _pattern in ('get_patches*.so', 'get_patches*.pyd'):
    for _f in file_path.parent.glob(_pattern):
        torch.ops.load_library(_f)
        _loaded = True

if not _loaded:
    raise FileNotFoundError(f'custom_ops shared library not found in {file_path.parent}')


class get_patches(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fmap, points, kernel_size):
        fmap = fmap.contiguous()
        points = points.contiguous()
        patches = torch.ops.custom_ops.get_patches_forward(fmap, points, kernel_size)

        ctx.save_for_backward(points, torch.tensor(fmap.shape))

        return patches

    @staticmethod
    def backward(ctx, d_patches):
        points, shape = ctx.saved_tensors
        H = shape[1].cpu().item()
        W = shape[2].cpu().item()
        d_fmap = torch.ops.custom_ops.get_patches_backward(d_patches.contiguous(), points, H, W)
        return d_fmap, None, None


def get_patches_torch(fmap: Tensor, points: Tensor, K: int):
    # fmap: CxHxW
    # points: Nx2
    N = points.shape[0]
    C = fmap.shape[0]
    radius = (K - 1.0) / 2.0
    pad_left_top = math.floor(radius)
    pad_right_bottom = math.ceil(radius)
    map_pad = F.pad(fmap.unsqueeze(0), (pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom)).squeeze(0)
    patches_left = (points[:, 1] - pad_left_top).long()
    patches_top = (points[:, 0] - pad_left_top).long()
    patches_right = patches_left + K
    patches_bottom = patches_top + K

    patches = map_pad[:, patches_top:patches_bottom, patches_left:patches_right]

    return patches
