"""
Validate LightGlue TRT engine by comparing with PyTorch LightGlue
using real image data from ALIKED.
"""
import os, sys, argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ALIKED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ALIKED')
sys.path.insert(0, os.path.abspath(ALIKED_DIR))

from nets.aliked import ALIKED
from lightglue import LightGlue

import tensorrt as trt


def trt_dtype_to_torch(dtype):
    if dtype == trt.DataType.FLOAT: return torch.float32
    elif dtype == trt.DataType.HALF: return torch.float16
    elif dtype == trt.DataType.INT32: return torch.int32
    elif dtype == trt.DataType.INT64: return torch.int64
    elif dtype == trt.DataType.INT8: return torch.int8
    elif dtype == trt.DataType.BOOL: return torch.bool
    return torch.float32


def run_trt_lightglue(engine_path, kpts0, desc0, kpts1, desc1):
    """Run LightGlue TRT engine using torch CUDA tensors. Returns matches0, mscores0 as numpy."""
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    N0, N1, D = kpts0.shape[1], kpts1.shape[1], desc0.shape[2]

    # Print IO tensor info
    print("  TRT IO tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"    {name}: mode={mode}, dtype={dtype}")

    # Set input shapes
    context.set_input_shape("kpts0", (1, N0, 2))
    context.set_input_shape("desc0", (1, N0, D))
    context.set_input_shape("kpts1", (1, N1, 2))
    context.set_input_shape("desc1", (1, N1, D))

    # Create torch CUDA tensors for inputs
    d_kpts0 = torch.from_numpy(kpts0).float().cuda().contiguous()
    d_desc0 = torch.from_numpy(desc0).float().cuda().contiguous()
    d_kpts1 = torch.from_numpy(kpts1).float().cuda().contiguous()
    d_desc1 = torch.from_numpy(desc1).float().cuda().contiguous()

    # Allocate output tensors
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        torch_dtype = trt_dtype_to_torch(dtype)
        t = torch.empty(shape, dtype=torch_dtype, device='cuda')
        buffers[name] = t
        print(f"    {name}: shape={shape}, torch_dtype={torch_dtype}")

    # Set tensor addresses
    context.set_tensor_address("kpts0", d_kpts0.data_ptr())
    context.set_tensor_address("desc0", d_desc0.data_ptr())
    context.set_tensor_address("kpts1", d_kpts1.data_ptr())
    context.set_tensor_address("desc1", d_desc1.data_ptr())
    context.set_tensor_address("matches0", buffers["matches0"].data_ptr())
    context.set_tensor_address("mscores0", buffers["mscores0"].data_ptr())

    # Run inference
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    print(f"  Inference ok: {ok}")

    matches0 = buffers["matches0"].cpu().numpy().flatten()
    mscores0 = buffers["mscores0"].cpu().numpy().flatten()
    return matches0, mscores0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image0', required=True)
    parser.add_argument('--image1', required=True)
    parser.add_argument('--engine', default='sfm-phoenix/engines/lightglue.engine')
    parser.add_argument('--max-edge', type=int, default=1600)
    parser.add_argument('--top-k', type=int, default=5000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Load ALIKED
    model = ALIKED(model_name='aliked-n32', device=args.device,
                   top_k=args.top_k, scores_th=0.2, n_limit=5000)
    model.eval()

    # Load images
    img0_bgr = cv2.imread(args.image0)
    img1_bgr = cv2.imread(args.image1)
    assert img0_bgr is not None and img1_bgr is not None

    def resize_max_edge(img, max_edge):
        h, w = img.shape[:2]
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        return img, scale

    img0_r, s0 = resize_max_edge(img0_bgr, args.max_edge)
    img1_r, s1 = resize_max_edge(img1_bgr, args.max_edge)
    print(f"Resized: {img0_r.shape[:2]}, {img1_r.shape[:2]}")

    # Extract features
    r0 = model.run(cv2.cvtColor(img0_r, cv2.COLOR_BGR2RGB))
    r1 = model.run(cv2.cvtColor(img1_r, cv2.COLOR_BGR2RGB))

    kpts0 = r0['keypoints']  # [N, 2] numpy
    kpts1 = r1['keypoints']
    desc0 = r0['descriptors']  # [N, D] numpy
    desc1 = r1['descriptors']

    print(f"Keypoints: {len(kpts0)}, {len(kpts1)}")
    print(f"kpts0 range: x=[{kpts0[:,0].min():.1f}, {kpts0[:,0].max():.1f}] y=[{kpts0[:,1].min():.1f}, {kpts0[:,1].max():.1f}]")
    print(f"desc0 first 5: {desc0[0,:5]}")
    print(f"desc0 norm: {np.linalg.norm(desc0[0]):.4f}")

    # ---- PyTorch LightGlue reference ----
    lg = LightGlue(features='aliked', n_layers=9, depth_confidence=-1, width_confidence=-1).to(args.device).eval()

    kpts0_t = torch.from_numpy(kpts0).float().unsqueeze(0).to(args.device)
    desc0_t = torch.from_numpy(desc0).float().unsqueeze(0).to(args.device)
    kpts1_t = torch.from_numpy(kpts1).float().unsqueeze(0).to(args.device)
    desc1_t = torch.from_numpy(desc1).float().unsqueeze(0).to(args.device)

    with torch.no_grad():
        lg_out = lg({'image0': {'keypoints': kpts0_t, 'descriptors': desc0_t},
                     'image1': {'keypoints': kpts1_t, 'descriptors': desc1_t}})

    matches_pt = lg_out['matches0'][0].cpu().numpy()
    mscores_pt = lg_out['matching_scores0'][0].cpu().numpy()
    n_matches_pt = (matches_pt >= 0).sum()
    print(f"\n[PyTorch] matches: {n_matches_pt}")
    print(f"[PyTorch] first 10 matches0: {matches_pt[:10]}")
    print(f"[PyTorch] first 10 mscores0: {np.array2string(mscores_pt[:10], precision=4)}")

    # ---- TRT Engine validation ----
    if os.path.exists(args.engine):
        print(f"\n--- TRT Engine Validation ---")
        kpts0_np = kpts0[np.newaxis].astype(np.float32)  # [1, N, 2]
        desc0_np = desc0[np.newaxis].astype(np.float32)
        kpts1_np = kpts1[np.newaxis].astype(np.float32)
        desc1_np = desc1[np.newaxis].astype(np.float32)

        matches_trt, mscores_trt = run_trt_lightglue(args.engine, kpts0_np, desc0_np, kpts1_np, desc1_np)

        n_matches_trt = (matches_trt >= 0).sum()
        print(f"\n[TRT] matches: {n_matches_trt}")
        print(f"[TRT] first 10 matches0: {matches_trt[:10]}")
        print(f"[TRT] first 10 mscores0: {np.array2string(mscores_trt[:10].astype(np.float32), precision=4)}")

        # Compare
        if n_matches_pt > 0 and n_matches_trt > 0:
            agree = (matches_pt == matches_trt.astype(matches_pt.dtype)).sum()
            print(f"\n[Comparison] PT={n_matches_pt}, TRT={n_matches_trt}, agree={agree}/{len(matches_pt)}")
        else:
            print(f"\n[Comparison] PT={n_matches_pt}, TRT={n_matches_trt}")
    else:
        print(f"Engine not found: {args.engine}")


if __name__ == '__main__':
    main()
