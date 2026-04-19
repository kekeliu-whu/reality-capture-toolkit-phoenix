"""Benchmark C++ TRT pipeline and report extraction / matching throughput."""
import argparse
import glob
import os
import re
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Benchmark C++ TRT pipeline')
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--n-pairs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of initial pairs to discard as warmup')
    parser.add_argument('--start', type=int, default=6)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--cpp-exe', default=None)
    parser.add_argument('--engine-dir', default='sfm-phoenix/engines')
    parser.add_argument('--max-edge', type=int, default=1600)
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    if not images:
        print(f"No JPG images found in {args.image_dir}")
        return

    def get_num(p):
        m = re.search(r'(\d+)\.jpg$', p)
        return int(m.group(1)) if m else 0
    images = sorted(images, key=get_num)
    nums = [get_num(p) for p in images]

    start_idx = nums.index(args.start) if args.start in nums else 0
    pairs = []
    idx = start_idx
    while len(pairs) < args.n_pairs and idx + args.step < len(images):
        pairs.append((images[idx], images[idx + args.step]))
        idx += args.step

    if len(pairs) < args.warmup + 1:
        print(f"Not enough pairs ({len(pairs)}) for warmup={args.warmup}")
        return

    print(f"Benchmarking {len(pairs)} pairs "
          f"({args.warmup} warmup + {len(pairs) - args.warmup} measured)")

    if args.cpp_exe:
        cpp_exe = args.cpp_exe
    else:
        cpp_exe = os.path.join('sfm-phoenix', 'build', 'Release',
                               'demo_feature_matching.exe')
    if not os.path.isfile(cpp_exe):
        print(f"C++ exe not found: {cpp_exe}")
        return

    engine_paths = {
        'backbone': os.path.join(args.engine_dir, 'aliked_backbone.engine'),
        'sddh': os.path.join(args.engine_dir, 'aliked_sddh.engine'),
        'lightglue': os.path.join(args.engine_dir, 'lightglue.engine'),
    }

    # Write pair list (no dump dirs → GPU-optimized path)
    with tempfile.TemporaryDirectory() as tmp:
        pair_list_path = os.path.join(tmp, 'pair_list.txt')
        with open(pair_list_path, 'w') as f:
            for i, (img0, img1) in enumerate(pairs):
                f.write(f"{img0} {img1}\n")

        cmd = [
            cpp_exe,
            '--backbone', engine_paths['backbone'],
            '--sddh', engine_paths['sddh'],
            '--lightglue', engine_paths['lightglue'],
            '--pair-list', pair_list_path,
            '--max-edge', str(args.max_edge),
        ]

        env = os.environ.copy()
        trt_bin = r'C:\Program Files\TensorRT-10.16.1.11\bin'
        env['PATH'] = trt_bin + ';' + env.get('PATH', '')
        env['PROFILE'] = '1'

        print("Running C++ pipeline...")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if proc.returncode != 0:
        print(f"FAILED:\n{proc.stderr}")
        return

    # Parse timing per pair — handles both GPU-optimized and legacy output.
    records = []
    cur = {}
    for line in proc.stdout.splitlines():
        if line.startswith('--- Pair '):
            if cur:
                records.append(cur)
            cur = {}
            continue
        # Legacy CPU path: "Image 0: 5000 keypoints (469.2 ms)"
        m = re.match(r'Image 0: (\d+) keypoints \(([\d.]+) ms\)', line)
        if m:
            cur['n0'] = int(m.group(1))
            cur['det0'] = float(m.group(2))
        m = re.match(r'Image 1: (\d+) keypoints \(([\d.]+) ms\)', line)
        if m:
            cur['n1'] = int(m.group(1))
            cur['det1'] = float(m.group(2))
        m = re.match(r'Matches: (\d+) \(([\d.]+) ms\)', line)
        if m:
            cur['matches'] = int(m.group(1))
            cur['match_ms'] = float(m.group(2))
        # GPU-optimized path: "Image 0: 5000 keypoints"
        m = re.match(r'Image 0: (\d+) keypoints$', line)
        if m:
            cur['n0'] = int(m.group(1))
        m = re.match(r'Image 1: (\d+) keypoints$', line)
        if m:
            cur['n1'] = int(m.group(1))
        # GPU-optimized: "Matches: 2563 (1100.5 ms total)"
        m = re.match(r'Matches: (\d+) \(([\d.]+) ms total\)', line)
        if m:
            cur['matches'] = int(m.group(1))
            cur['total_ms'] = float(m.group(2))
    if cur:
        records.append(cur)

    # Parse PROFILE lines from stderr. In GPU batch mode, steady-state pairs use
    # the cached-left-image path, so per-pair extraction is det1 only.
    profile_records = []
    for line in proc.stderr.splitlines():
        line = line.strip()
        if '[PROFILE]' not in line:
            continue

        cached = 'det0=cached' in line
        if cached:
            match = re.search(
                r'det0=cached det1=([\d.]+)ms match=([\d.]+)ms '
                r'total=([\d.]+)ms', line)
            if not match:
                continue
            profile_records.append({
                'cached': True,
                'det1_profile_ms': float(match.group(1)),
                'match_profile_ms': float(match.group(2)),
                'total_profile_ms': float(match.group(3)),
            })
            continue

        match = re.search(
            r'det0=([\d.]+)ms det1=([\d.]+)ms match=([\d.]+)ms '
            r'total=([\d.]+)ms', line)
        if not match:
            continue
        profile_records.append({
            'cached': False,
            'det0_profile_ms': float(match.group(1)),
            'det1_profile_ms': float(match.group(2)),
            'match_profile_ms': float(match.group(3)),
            'total_profile_ms': float(match.group(4)),
        })

    for idx, prof in enumerate(profile_records):
        if idx >= len(records):
            break
        records[idx].update(prof)

    if len(records) < args.warmup + 1:
        print(f"Only {len(records)} pairs parsed, need > {args.warmup}")
        return

    # Discard warmup
    measured = records[args.warmup:]
    n = len(measured)
    print(f"\nDiscarded {args.warmup} warmup pairs, {n} measured pairs\n")

    # Detect whether GPU-optimized or legacy format
    has_split = 'det0' in measured[0]
    has_total = 'total_ms' in measured[0]
    has_profile = 'match_profile_ms' in measured[0]

    n0 = np.array([r.get('n0', 0) for r in measured])
    n1 = np.array([r.get('n1', 0) for r in measured])
    matches = np.array([r.get('matches', 0) for r in measured])

    def stats(arr):
        return (np.mean(arr), np.std(arr), np.min(arr),
                np.median(arr), np.max(arr))

    def fmt_stats(arr, unit='ms'):
        mean, std, mn, med, mx = stats(arr)
        return (f"mean={mean:7.1f}{unit}  std={std:5.1f}{unit}  "
                f"min={mn:7.1f}  med={med:7.1f}  max={mx:7.1f}")

    print("=" * 72)
    print("C++ TRT Pipeline Benchmark (FP32)")
    print(f"  Images: {os.path.basename(args.image_dir)}")
    print(f"  Pairs: {n} (after {args.warmup} warmup)")
    print(f"  Keypoints: {np.mean(n0):.0f} / {np.mean(n1):.0f} avg")
    print(f"  Matches: {np.mean(matches):.0f} avg")
    print("=" * 72)
    print()

    if has_profile:
        extract_ms = np.array([r['det1_profile_ms'] for r in measured])
        match_ms = np.array([r['match_profile_ms'] for r in measured])
        total = np.array([r['total_profile_ms'] for r in measured])
        print(f"{'Extraction (per image)':>25s}:  {fmt_stats(extract_ms)}")
        print(f"{'Matching (per pair)':>25s}:  {fmt_stats(match_ms)}")
        print(f"{'Total (per pair)':>25s}:  {fmt_stats(total)}")
    elif has_split:
        det0 = np.array([r['det0'] for r in measured])
        det1 = np.array([r['det1'] for r in measured])
        det_all = np.concatenate([det0, det1])
        match_ms = np.array([r['match_ms'] for r in measured])
        total = det0 + det1 + match_ms
        print(f"{'Detection (per image)':>25s}:  {fmt_stats(det_all)}")
        print(f"{'  Det img0':>25s}:  {fmt_stats(det0)}")
        print(f"{'  Det img1':>25s}:  {fmt_stats(det1)}")
        print(f"{'Matching (per pair)':>25s}:  {fmt_stats(match_ms)}")
        print(f"{'Total (per pair)':>25s}:  {fmt_stats(total)}")
    elif has_total:
        total = np.array([r['total_ms'] for r in measured])
        print(f"  (GPU-optimized path — combined timing)")
        print(f"{'Total (per pair)':>25s}:  {fmt_stats(total)}")
    else:
        print("ERROR: Could not parse timing data")
        return
    print()

    # FPS
    mean_total = np.mean(total)
    pair_tps = 1000.0 / mean_total if mean_total > 0 else 0
    print(f"Pair throughput: {pair_tps:.2f} pairs/sec  "
          f"({mean_total:.1f} ms/pair)")

    if has_profile:
        mean_extract = np.mean(extract_ms)
        mean_match = np.mean(match_ms)
        extract_tps = 1000.0 / mean_extract if mean_extract > 0 else 0
        match_tps = 1000.0 / mean_match if mean_match > 0 else 0
        print(f"Extract TPS: {extract_tps:.2f} images/sec  "
              f"({mean_extract:.2f} ms/image)")
        print(f"Match TPS:   {match_tps:.2f} pairs/sec  "
              f"({mean_match:.2f} ms/pair)")
    print()

    # Per-pair table
    if has_profile:
        hdr = (f"{'#':>4s} | {'Extract':>8s} {'Match':>8s} {'Total':>8s} | "
               f"{'Kpts0':>5s} {'Kpts1':>5s} {'Matches':>7s} {'Cached':>6s}")
        print(hdr)
        print('-' * len(hdr))
        for i, r in enumerate(measured):
            print(f"{i:>4d} | {r['det1_profile_ms']:>8.1f} "
                  f"{r['match_profile_ms']:>8.1f} "
                  f"{r['total_profile_ms']:>8.1f} | "
                  f"{r.get('n0',0):>5d} {r.get('n1',0):>5d} "
                  f"{r.get('matches',0):>7d} "
                  f"{str(r.get('cached', False)):>6s}")
    elif has_split:
        hdr = (f"{'#':>4s} | {'Det0':>8s} {'Det1':>8s} {'Match':>8s} "
               f"{'Total':>8s} | {'Kpts0':>5s} {'Kpts1':>5s} {'Matches':>7s}")
        print(hdr)
        print('-' * len(hdr))
        for i, r in enumerate(measured):
            t = r['det0'] + r['det1'] + r['match_ms']
            print(f"{i:>4d} | {r['det0']:>8.1f} {r['det1']:>8.1f} "
                  f"{r['match_ms']:>8.1f} {t:>8.1f} | "
                  f"{r['n0']:>5d} {r['n1']:>5d} {r['matches']:>7d}")
    elif has_total:
        hdr = (f"{'#':>4s} | {'Total':>8s} | "
               f"{'Kpts0':>5s} {'Kpts1':>5s} {'Matches':>7s}")
        print(hdr)
        print('-' * len(hdr))
        for i, r in enumerate(measured):
            print(f"{i:>4d} | {r['total_ms']:>8.1f} | "
                  f"{r.get('n0',0):>5d} {r.get('n1',0):>5d} "
                  f"{r.get('matches',0):>7d}")


if __name__ == '__main__':
    main()
