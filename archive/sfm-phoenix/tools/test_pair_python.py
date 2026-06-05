"""
Run ALIKED + LightGlue on an image pair (Python reference) for comparison with C++ TRT.

Usage:
    cd ALIKED
    python ../sfm-phoenix/tools/test_pair_python.py \
        --image0 path/to/img0.jpg --image1 path/to/img1.jpg \
        --output matches_py.jpg
"""
import os, sys, time, argparse
import cv2
import torch
import numpy as np

ALIKED_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw', 'ALIKED')
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.abspath(ALIKED_DIR))

from nets.aliked import ALIKED

try:
    from lightglue import LightGlue
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image0', required=True)
    parser.add_argument('--image1', required=True)
    parser.add_argument('--output', default='matches_py.jpg')
    parser.add_argument('--model', default='aliked-n32')
    parser.add_argument('--max-edge', type=int, default=1600)
    parser.add_argument('--top-k', type=int, default=5000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    assert LIGHTGLUE_AVAILABLE, "LightGlue not installed"

    # Load model
    model = ALIKED(model_name=args.model, device=args.device,
                   top_k=args.top_k, scores_th=0.2, n_limit=5000)
    model.eval()

    lg = LightGlue(features='aliked', n_layers=9, depth_confidence=-1, width_confidence=-1).to(args.device).eval()

    # Load images
    img0_bgr = cv2.imread(args.image0)
    img1_bgr = cv2.imread(args.image1)
    assert img0_bgr is not None and img1_bgr is not None

    # Resize (same as C++ max_edge)
    def resize_max_edge(img, max_edge):
        h, w = img.shape[:2]
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        return img, scale

    img0_r, s0 = resize_max_edge(img0_bgr, args.max_edge)
    img1_r, s1 = resize_max_edge(img1_bgr, args.max_edge)

    # model.run() takes BGR, internally converts and processes
    # Actually run() takes RGB tensor, let's use forward() via run()
    # The run() method expects HWC numpy or converts via ToTensor
    # Let's use model.run() which handles everything

    # Warmup
    r0_warmup = model.run(cv2.cvtColor(img0_r, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize()

    # Extract features
    t_start = time.perf_counter()
    r0 = model.run(cv2.cvtColor(img0_r, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize()
    t_det0 = time.perf_counter()

    r1 = model.run(cv2.cvtColor(img1_r, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize()
    t_det1 = time.perf_counter()

    kpts0 = r0['keypoints']  # [N, 2] numpy, pixel coords
    kpts1 = r1['keypoints']
    desc0 = r0['descriptors']  # [N, D] numpy
    desc1 = r1['descriptors']
    scores0 = r0['scores']
    scores1 = r1['scores']

    n0 = len(kpts0)
    n1 = len(kpts1)
    print(f"Image 0: {n0} keypoints ({(t_det0 - t_start)*1000:.1f} ms)")
    print(f"Image 1: {n1} keypoints ({(t_det1 - t_det0)*1000:.1f} ms)")

    # Match with LightGlue (expects pixel coords as (1, N, 2) tensors)
    kpts0_t = torch.from_numpy(kpts0).float().unsqueeze(0).to(args.device)
    desc0_t = torch.from_numpy(desc0).float().unsqueeze(0).to(args.device)
    kpts1_t = torch.from_numpy(kpts1).float().unsqueeze(0).to(args.device)
    desc1_t = torch.from_numpy(desc1).float().unsqueeze(0).to(args.device)

    lg_input = {
        'image0': {'keypoints': kpts0_t, 'descriptors': desc0_t},
        'image1': {'keypoints': kpts1_t, 'descriptors': desc1_t},
    }

    t_match_start = time.perf_counter()
    with torch.no_grad():
        lg_out = lg(lg_input)
    torch.cuda.synchronize()
    t_match_end = time.perf_counter()

    # Extract matches (matches0: [1, N0] index into kpts1, -1 = no match)
    matches0 = lg_out['matches0'][0].cpu().numpy()  # [N0]
    valid = matches0 >= 0
    m0_idx = np.where(valid)[0]
    m1_idx = matches0[valid].astype(int)

    print(f"Matches: {len(m0_idx)} ({(t_match_end - t_match_start)*1000:.1f} ms)")

    # Draw matches
    h0, w0 = img0_r.shape[:2]
    h1, w1 = img1_r.shape[:2]
    H = max(h0, h1)
    W = w0 + w1
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    canvas[:h0, :w0] = img0_r
    canvas[:h1, w0:w0+w1] = img1_r

    for kp in kpts0:
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1, cv2.LINE_AA)
    for kp in kpts1:
        cv2.circle(canvas, (int(kp[0]) + w0, int(kp[1])), 2, (0, 0, 255), -1, cv2.LINE_AA)

    for i, j in zip(m0_idx, m1_idx):
        p0 = (int(kpts0[i, 0]), int(kpts0[i, 1]))
        p1 = (int(kpts1[j, 0]) + w0, int(kpts1[j, 1]))
        cv2.line(canvas, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(args.output, canvas)
    print(f"Saved: {args.output}")

    # Print some keypoint stats for comparison
    print(f"\nTop-5 keypoints image0 (x, y, score):")
    idx = np.argsort(-scores0)[:5]
    for i in idx:
        print(f"  ({kpts0[i,0]:.1f}, {kpts0[i,1]:.1f}) score={scores0[i]:.4f}")


if __name__ == '__main__':
    main()
