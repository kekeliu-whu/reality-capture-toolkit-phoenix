"""
Compare Python (ALIKED + LightGlue) vs C++ TRT pipeline on sequential image pairs.

Outputs per-pair and aggregate statistics for:
  - Keypoint count
  - Keypoint position overlap (nearest-neighbor within threshold)
  - Descriptor cosine similarity (matched keypoints)
  - Match count
  - Match overlap (identical (i,j) pairs in both pipelines)
  - Timing

Usage:
    cd <repo_root>
    python feature_extraction/tools/compare_py_cpp.py \
        --image-dir <dir_with_numbered_jpgs> \
        --n-pairs 10 \
        [--start 6] \
        [--step 1] \
        [--output-dir feature_extraction/compare_output]
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Python ALIKED + LightGlue
# ---------------------------------------------------------------------------
ALIKED_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw', 'ALIKED')
# Ensure repo root is on sys.path for absolute imports in ALIKED code
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.abspath(ALIKED_DIR))


def load_python_models(device='cuda'):
    import torch
    from feature_extraction.raw.ALIKED.nets.aliked import ALIKED
    from lightglue import LightGlue

    model = ALIKED(model_name='aliked-n32', device=device,
                   top_k=5000, scores_th=0.2, n_limit=5000)
    model.eval()

    lg = LightGlue(features='aliked', n_layers=9,
                    depth_confidence=-1, width_confidence=-1)
    lg = lg.to(device).eval()
    return model, lg, device


def run_python_pair(model, lg, device, img0_bgr, img1_bgr, max_edge=1600):
    """Run Python ALIKED + LightGlue and return dict with results."""
    import torch

    def resize_max_edge(img, me):
        h, w = img.shape[:2]
        s = min(1.0, me / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_AREA)
        return img, s

    img0_r, s0 = resize_max_edge(img0_bgr, max_edge)
    img1_r, s1 = resize_max_edge(img1_bgr, max_edge)

    t0 = time.perf_counter()
    r0 = model.run(cv2.cvtColor(img0_r, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    r1 = model.run(cv2.cvtColor(img1_r, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    kpts0 = r0['keypoints']  # [N, 2]
    kpts1 = r1['keypoints']
    desc0 = r0['descriptors']  # [N, D]
    desc1 = r1['descriptors']
    scores0 = r0['scores']
    scores1 = r1['scores']

    # Match with LightGlue
    kpts0_t = torch.from_numpy(kpts0).float().unsqueeze(0).to(device)
    desc0_t = torch.from_numpy(desc0).float().unsqueeze(0).to(device)
    kpts1_t = torch.from_numpy(kpts1).float().unsqueeze(0).to(device)
    desc1_t = torch.from_numpy(desc1).float().unsqueeze(0).to(device)

    lg_input = {
        'image0': {'keypoints': kpts0_t, 'descriptors': desc0_t},
        'image1': {'keypoints': kpts1_t, 'descriptors': desc1_t},
    }
    t_m0 = time.perf_counter()
    with torch.no_grad():
        lg_out = lg(lg_input)
    torch.cuda.synchronize()
    t_m1 = time.perf_counter()

    matches0 = lg_out['matches0'][0].cpu().numpy()
    valid = matches0 >= 0
    m0_idx = np.where(valid)[0]
    m1_idx = matches0[valid].astype(int)
    mscores = lg_out['matching_scores0'][0].cpu().numpy()[valid]

    return {
        'kpts0': kpts0, 'kpts1': kpts1,
        'desc0': desc0, 'desc1': desc1,
        'scores0': scores0, 'scores1': scores1,
        'match_pairs': np.stack([m0_idx, m1_idx], axis=1) if len(m0_idx) > 0 else np.zeros((0, 2), dtype=int),
        'mscores': mscores,
        'n0': len(kpts0), 'n1': len(kpts1),
        'n_matches': len(m0_idx),
        'time_det0_ms': (t1 - t0) * 1000,
        'time_det1_ms': (t2 - t1) * 1000,
        'time_match_ms': (t_m1 - t_m0) * 1000,
    }


# ---------------------------------------------------------------------------
# C++ TRT pipeline (batch mode: single process, loads engines once)
# ---------------------------------------------------------------------------
def run_cpp_batch(exe_path, engine_paths, pairs, output_dir):
    """Run C++ in batch mode via --pair-list. Returns list of result dicts."""
    pair_list_path = os.path.join(output_dir, 'pair_list.txt')
    dump_dirs = []
    with open(pair_list_path, 'w') as f:
        for idx, (img0, img1) in enumerate(pairs):
            dd = os.path.join(output_dir, f'cpp_dump_{idx:04d}')
            os.makedirs(dd, exist_ok=True)
            dump_dirs.append(dd)
            f.write(f"{img0} {img1} {dd}\n")

    cmd = [
        exe_path,
        '--backbone', engine_paths['backbone'],
        '--sddh', engine_paths['sddh'],
        '--lightglue', engine_paths['lightglue'],
        '--pair-list', pair_list_path,
    ]

    env = os.environ.copy()
    trt_bin = r'C:\Program Files\TensorRT-10.16.1.11\bin'
    env['PATH'] = trt_bin + ';' + env.get('PATH', '')

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"C++ batch FAILED:\n{proc.stderr}")
        return None

    # Parse stdout per pair
    stdout_lines = proc.stdout.splitlines()
    results = []
    pair_idx = -1
    cur = {}

    for line in stdout_lines:
        if line.startswith('--- Pair '):
            if cur:
                results.append(cur)
            pair_idx += 1
            cur = {'pair_idx': pair_idx}
            continue
        m = re.match(r'Image 0: (\d+) keypoints \(([\d.]+) ms\)', line)
        if m:
            cur['n0'] = int(m.group(1))
            cur['time_det0_ms'] = float(m.group(2))
        m = re.match(r'Image 1: (\d+) keypoints \(([\d.]+) ms\)', line)
        if m:
            cur['n1'] = int(m.group(1))
            cur['time_det1_ms'] = float(m.group(2))
        m = re.match(r'Matches: (\d+) \(([\d.]+) ms\)', line)
        if m:
            cur['n_matches'] = int(m.group(1))
            cur['time_match_ms'] = float(m.group(2))
    if cur:
        results.append(cur)

    # Load binary dumps for each pair
    for i, r in enumerate(results):
        dd = dump_dirs[i]
        n0 = r.get('n0', 0)
        n1 = r.get('n1', 0)
        nm = r.get('n_matches', 0)

        r['kpts0'] = np.fromfile(os.path.join(dd, 'kpts0.bin'), np.float32).reshape(-1, 2)
        r['kpts1'] = np.fromfile(os.path.join(dd, 'kpts1.bin'), np.float32).reshape(-1, 2)
        r['scores0'] = np.fromfile(os.path.join(dd, 'scores0.bin'), np.float32)
        r['scores1'] = np.fromfile(os.path.join(dd, 'scores1.bin'), np.float32)
        r['desc0'] = np.fromfile(os.path.join(dd, 'desc0.bin'), np.float32).reshape(n0, -1)
        r['desc1'] = np.fromfile(os.path.join(dd, 'desc1.bin'), np.float32).reshape(n1, -1)

        if nm > 0:
            r['match_pairs'] = np.fromfile(os.path.join(dd, 'matches.bin'), np.int32).reshape(-1, 2)
            r['mscores'] = np.fromfile(os.path.join(dd, 'mscores.bin'), np.float32)
        else:
            r['match_pairs'] = np.zeros((0, 2), dtype=np.int32)
            r['mscores'] = np.zeros(0, dtype=np.float32)

    return results


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
def compare_keypoints(py_kpts, cpp_kpts, threshold=3.0):
    """
    Compare two keypoint sets. For each Python kpt, find the nearest C++ kpt.
    Returns: fraction of Python kpts that have a C++ neighbor within threshold.
    """
    from scipy.spatial import cKDTree
    if len(py_kpts) == 0 or len(cpp_kpts) == 0:
        return 0.0, float('inf')
    tree = cKDTree(cpp_kpts)
    dists, _ = tree.query(py_kpts, k=1)
    within = np.sum(dists < threshold)
    return within / len(py_kpts), float(np.median(dists))


def compare_descriptors_at_matched_kpts(py_kpts, py_desc, cpp_kpts, cpp_desc,
                                         kpt_threshold=3.0):
    """
    For Python keypoints that have a nearby C++ keypoint (within threshold),
    compute cosine similarity of their descriptors.
    """
    from scipy.spatial import cKDTree
    if len(py_kpts) == 0 or len(cpp_kpts) == 0:
        return 0.0, 0
    tree = cKDTree(cpp_kpts)
    dists, indices = tree.query(py_kpts, k=1)
    mask = dists < kpt_threshold
    if mask.sum() == 0:
        return 0.0, 0

    py_d = py_desc[mask]
    cpp_d = cpp_desc[indices[mask]]
    # Normalize
    py_norm = py_d / (np.linalg.norm(py_d, axis=1, keepdims=True) + 1e-8)
    cpp_norm = cpp_d / (np.linalg.norm(cpp_d, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(py_norm * cpp_norm, axis=1)
    return float(np.mean(cos_sim)), int(mask.sum())


def compare_matches(py_pairs, cpp_pairs):
    """
    Compare match sets. Returns: num common pairs (same (i0, i1)).
    Since keypoint orderings differ, we compare by spatial proximity instead.
    """
    # Direct index comparison only works if keypoint ordering is the same.
    # Since DKD may produce differently ordered keypoints, we just compare counts.
    return {
        'py_matches': len(py_pairs),
        'cpp_matches': len(cpp_pairs),
    }


def compare_match_overlap_spatial(py_kpts0, py_kpts1, py_pairs,
                                   cpp_kpts0, cpp_kpts1, cpp_pairs,
                                   kpt_threshold=5.0):
    """
    For each Python match (p0→p1), check if there's a C++ match whose
    endpoints are both within kpt_threshold pixels. Returns overlap ratio.
    """
    if len(py_pairs) == 0 or len(cpp_pairs) == 0:
        return 0.0

    # Build arrays of match endpoints in image space
    py_pts0 = py_kpts0[py_pairs[:, 0]]
    py_pts1 = py_kpts1[py_pairs[:, 1]]
    cpp_pts0 = cpp_kpts0[cpp_pairs[:, 0]]
    cpp_pts1 = cpp_kpts1[cpp_pairs[:, 1]]

    # For each Python match, find if any C++ match has both endpoints close
    from scipy.spatial import cKDTree
    tree0 = cKDTree(cpp_pts0)
    tree1 = cKDTree(cpp_pts1)

    count = 0
    # Batch: for each py match, find closest cpp match by endpoint0
    d0, idx0 = tree0.query(py_pts0, k=1)
    for i in range(len(py_pairs)):
        if d0[i] < kpt_threshold:
            # Check if the matched C++ pair's endpoint1 is also close
            d1 = np.linalg.norm(cpp_pts1[idx0[i]] - py_pts1[i])
            if d1 < kpt_threshold:
                count += 1

    return count / len(py_pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Compare Python vs C++ TRT feature extraction pipeline')
    parser.add_argument('--image-dir', required=True,
                        help='Directory with numbered JPG images')
    parser.add_argument('--n-pairs', type=int, default=10,
                        help='Number of consecutive pairs to test')
    parser.add_argument('--start', type=int, default=None,
                        help='Starting image number (default: first found)')
    parser.add_argument('--step', type=int, default=1,
                        help='Step between consecutive frames')
    parser.add_argument('--output-dir', default='feature_extraction/compare_output',
                        help='Output directory for results')
    parser.add_argument('--cpp-exe', default=None,
                        help='Path to demo_feature_matching.exe')
    parser.add_argument('--engine-dir', default='feature_extraction/engines',
                        help='Directory containing TRT engine files')
    parser.add_argument('--lightglue-engine', default=None,
                        help='Override LightGlue engine path (e.g. lightglue_fp16.engine)')
    parser.add_argument('--max-edge', type=int, default=1600)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Discover images
    images = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    if not images:
        print(f"No JPG images found in {args.image_dir}")
        return

    # Extract numbers and sort
    def get_num(p):
        m = re.search(r'(\d+)\.jpg$', p)
        return int(m.group(1)) if m else 0
    images = sorted(images, key=get_num)
    nums = [get_num(p) for p in images]
    num_to_path = dict(zip(nums, images))

    if args.start is not None:
        start_idx = nums.index(args.start) if args.start in nums else 0
    else:
        start_idx = 0

    # Build pairs
    pairs = []
    idx = start_idx
    while len(pairs) < args.n_pairs and idx + args.step < len(images):
        pairs.append((images[idx], images[idx + args.step]))
        idx += args.step

    if not pairs:
        print("Not enough images for requested pairs")
        return

    print(f"Testing {len(pairs)} image pairs from {args.image_dir}")
    print(f"  First pair: {os.path.basename(pairs[0][0])} <-> {os.path.basename(pairs[0][1])}")
    print(f"  Last pair:  {os.path.basename(pairs[-1][0])} <-> {os.path.basename(pairs[-1][1])}")
    print()

    # Find C++ exe
    if args.cpp_exe:
        cpp_exe = args.cpp_exe
    else:
        cpp_exe = os.path.join('feature_extraction', 'build', 'Release',
                               'demo_feature_matching.exe')
    if not os.path.isfile(cpp_exe):
        print(f"C++ exe not found: {cpp_exe}")
        return

    lg_engine = (args.lightglue_engine
                  if args.lightglue_engine
                  else os.path.join(args.engine_dir, 'lightglue.engine'))
    engine_paths = {
        'backbone': os.path.join(args.engine_dir, 'aliked_backbone.engine'),
        'sddh': os.path.join(args.engine_dir, 'aliked_sddh.engine'),
        'lightglue': lg_engine,
    }
    for k, v in engine_paths.items():
        if not os.path.isfile(v):
            print(f"Engine not found: {v}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Python models
    print("Loading Python models...")
    model, lg, device = load_python_models(args.device)

    # Warmup Python (1 pair)
    print("Warming up Python pipeline...")
    img_warmup = cv2.imread(pairs[0][0])
    import torch
    _ = model.run(cv2.cvtColor(
        cv2.resize(img_warmup, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize()

    # --- Run all Python pairs ---
    print("Running Python pipeline on all pairs...")
    py_results = []
    for pair_idx, (img0_path, img1_path) in enumerate(pairs):
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        if img0 is None or img1 is None:
            py_results.append(None)
            continue
        py = run_python_pair(model, lg, device, img0, img1, args.max_edge)
        py_results.append(py)
        print(f"  Pair {pair_idx}: Py kpts={py['n0']}/{py['n1']}, "
              f"matches={py['n_matches']}")

    # --- Run all C++ pairs in batch ---
    print("\nRunning C++ pipeline in batch mode...")
    cpp_results = run_cpp_batch(cpp_exe, engine_paths, pairs, args.output_dir)
    if cpp_results is None:
        print("C++ batch failed!")
        return

    # --- Compare ---
    results = []
    print(f"\n{'='*80}")
    print(f"{'Pair':>20s} | {'Py kpts':>8s} {'C++ kpts':>9s} | "
          f"{'KptOvlp':>7s} {'MedDist':>7s} | {'DescSim':>7s} | "
          f"{'Py match':>8s} {'C++ match':>9s} {'MtchOvlp':>8s} | "
          f"{'Py ms':>7s} {'C++ ms':>7s}")
    print(f"{'-'*80}")

    for pair_idx, (img0_path, img1_path) in enumerate(pairs):
        name0 = os.path.splitext(os.path.basename(img0_path))[0]
        name1 = os.path.splitext(os.path.basename(img1_path))[0]
        pair_name = f"{name0}-{name1}"

        py = py_results[pair_idx]
        if py is None:
            print(f"  {pair_name}: Python failed")
            continue
        if pair_idx >= len(cpp_results):
            print(f"  {pair_name}: C++ result missing")
            continue
        cpp = cpp_results[pair_idx]

        # Compare keypoints
        kpt_overlap0, med_dist0 = compare_keypoints(py['kpts0'], cpp['kpts0'])
        kpt_overlap1, med_dist1 = compare_keypoints(py['kpts1'], cpp['kpts1'])
        kpt_overlap = (kpt_overlap0 + kpt_overlap1) / 2
        med_dist = (med_dist0 + med_dist1) / 2

        # Compare descriptors at spatially matched keypoints
        desc_sim0, n_matched0 = compare_descriptors_at_matched_kpts(
            py['kpts0'], py['desc0'], cpp['kpts0'], cpp['desc0'])
        desc_sim1, n_matched1 = compare_descriptors_at_matched_kpts(
            py['kpts1'], py['desc1'], cpp['kpts1'], cpp['desc1'])
        desc_sim = (desc_sim0 + desc_sim1) / 2 if (n_matched0 + n_matched1) > 0 else 0

        # Compare matches (spatial overlap)
        match_overlap = compare_match_overlap_spatial(
            py['kpts0'], py['kpts1'], py['match_pairs'],
            cpp['kpts0'], cpp['kpts1'], cpp['match_pairs'])

        py_total_ms = py['time_det0_ms'] + py['time_det1_ms'] + py['time_match_ms']
        cpp_total_ms = cpp['time_det0_ms'] + cpp['time_det1_ms'] + cpp['time_match_ms']

        row = {
            'pair': pair_name,
            'py_n0': py['n0'], 'py_n1': py['n1'],
            'cpp_n0': cpp['n0'], 'cpp_n1': cpp['n1'],
            'kpt_overlap': kpt_overlap,
            'kpt_median_dist': med_dist,
            'desc_cosine_sim': desc_sim,
            'desc_n_compared': n_matched0 + n_matched1,
            'py_matches': py['n_matches'],
            'cpp_matches': cpp['n_matches'],
            'match_overlap': match_overlap,
            'py_total_ms': py_total_ms,
            'cpp_total_ms': cpp_total_ms,
            'py_det0_ms': py['time_det0_ms'],
            'py_det1_ms': py['time_det1_ms'],
            'py_match_ms': py['time_match_ms'],
            'cpp_det0_ms': cpp['time_det0_ms'],
            'cpp_det1_ms': cpp['time_det1_ms'],
            'cpp_match_ms': cpp['time_match_ms'],
        }
        results.append(row)

        avg_kpts_py = (py['n0'] + py['n1']) / 2
        avg_kpts_cpp = (cpp['n0'] + cpp['n1']) / 2
        print(f"{pair_name:>20s} | {avg_kpts_py:>8.0f} {avg_kpts_cpp:>9.0f} | "
              f"{kpt_overlap:>7.1%} {med_dist:>7.1f} | {desc_sim:>7.4f} | "
              f"{py['n_matches']:>8d} {cpp['n_matches']:>9d} {match_overlap:>8.1%} | "
              f"{py_total_ms:>7.0f} {cpp_total_ms:>7.0f}")

    # Aggregate
    if results:
        print(f"{'='*80}")
        n = len(results)
        avg = lambda key: sum(r[key] for r in results) / n

        print(f"\n{'SUMMARY':>20s}   (over {n} pairs)")
        print(f"  Avg keypoints:      Py={avg('py_n0'):.0f}/{avg('py_n1'):.0f}  "
              f"C++={avg('cpp_n0'):.0f}/{avg('cpp_n1'):.0f}")
        print(f"  Keypoint overlap:   {avg('kpt_overlap'):.1%}  "
              f"(median dist: {avg('kpt_median_dist'):.2f} px)")
        print(f"  Descriptor cosine:  {avg('desc_cosine_sim'):.4f}")
        print(f"  Matches:            Py={avg('py_matches'):.0f}  "
              f"C++={avg('cpp_matches'):.0f}")
        print(f"  Match overlap:      {avg('match_overlap'):.1%}")
        print(f"  Timing (total):     Py={avg('py_total_ms'):.0f}ms  "
              f"C++={avg('cpp_total_ms'):.0f}ms")
        print(f"    Detection:        Py={avg('py_det0_ms'):.0f}+{avg('py_det1_ms'):.0f}ms  "
              f"C++={avg('cpp_det0_ms'):.0f}+{avg('cpp_det1_ms'):.0f}ms")
        print(f"    Matching:         Py={avg('py_match_ms'):.0f}ms  "
              f"C++={avg('cpp_match_ms'):.0f}ms")

        # Save JSON
        json_path = os.path.join(args.output_dir, 'comparison_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {json_path}")


if __name__ == '__main__':
    main()
