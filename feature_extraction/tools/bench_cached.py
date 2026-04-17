"""Benchmark C++ TRT pipeline with detection caching + FP16 LightGlue."""
import subprocess, tempfile, os, re, sys
import numpy as np
import glob

imgdir = sys.argv[1] if len(sys.argv) > 1 else r'D:\ProjectX\project-3d\data\sfm\external-cameras\hkustgz\xsfm_output\ground_undistort\fisheye_x5_VID_20251017_113930_00_052_cam0'
n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 22
warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 2
lg_engine = sys.argv[4] if len(sys.argv) > 4 else 'feature_extraction/engines/lightglue_fp16.engine'

images = sorted(glob.glob(os.path.join(imgdir, '*.jpg')),
                key=lambda p: int(re.search(r'(\d+)\.jpg$', p).group(1)))
start_idx = next(i for i, p in enumerate(images) if '000006.jpg' in p)
pairs = [(images[start_idx + i], images[start_idx + i + 1])
         for i in range(n_pairs)]

exe = r'feature_extraction\build\Release\demo_feature_matching.exe'

with tempfile.TemporaryDirectory() as tmp:
    pf = os.path.join(tmp, 'pairs.txt')
    with open(pf, 'w') as f:
        for a, b in pairs:
            f.write(f'{a} {b}\n')
    cmd = [exe,
           '--backbone', 'feature_extraction/engines/aliked_backbone.engine',
           '--sddh', 'feature_extraction/engines/aliked_sddh.engine',
           '--lightglue', lg_engine,
           '--pair-list', pf]
    env = os.environ.copy()
    env['PATH'] = r'C:\Program Files\TensorRT-10.16.1.11\bin;' + env.get('PATH', '')
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

records = []
cur = {}
for line in proc.stdout.splitlines():
    if line.startswith('--- Pair '):
        if cur:
            records.append(cur)
        cur = {}
    m = re.match(r'Matches: (\d+) \(([\d.]+) ms total\)', line)
    if m:
        cur['matches'] = int(m.group(1))
        cur['total_ms'] = float(m.group(2))
    m = re.match(r'Image 0: (\d+) keypoints', line)
    if m:
        cur['n0'] = int(m.group(1))
    m = re.match(r'Image 1: (\d+) keypoints', line)
    if m:
        cur['n1'] = int(m.group(1))
if cur:
    records.append(cur)

measured = records[warmup:]
if not measured:
    print("No measured pairs!")
    sys.exit(1)

total = np.array([r['total_ms'] for r in measured])
matches = np.array([r.get('matches', 0) for r in measured])

lg_name = os.path.basename(lg_engine)
print(f'=== C++ TRT Pipeline ({lg_name}, detection caching) ===')
print(f'Pairs: {len(measured)} (after {warmup} warmup)')
print(f'Matches: {np.mean(matches):.0f} avg')
print(f'Total: mean={np.mean(total):.1f}ms  std={np.std(total):.1f}ms  '
      f'min={np.min(total):.1f}ms  max={np.max(total):.1f}ms')
print(f'Throughput: {1000/np.mean(total):.2f} pairs/sec '
      f'({np.mean(total):.1f} ms/pair)')
print()
for i, r in enumerate(measured):
    print(f'  {i:>2d}: {r["total_ms"]:>7.1f}ms  matches={r.get("matches", 0)}')
