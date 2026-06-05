"""Benchmark ALIKED latency under TensorRT and ORT CUDA.

This script compares the unified ALIKED ONNX model in two runtime backends:
  1. TensorRT via a prebuilt engine and trtexec
  2. ONNX Runtime CUDAExecutionProvider

Both paths keep tensors on GPU to avoid host transfer noise, which better
matches Phoenix's extractor pipeline.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def DefaultBuildDir() -> Path:
  return Path(__file__).resolve().parents[1] / "build" / "RelWithDebInfo"


def ParseShape(text: str) -> tuple[int, ...]:
  parts = [part for part in re.split(r"[x,]", text.strip()) if part]
  if not parts:
    raise ValueError(f"Invalid shape: {text}")
  return tuple(int(part) for part in parts)


def ShapeText(shape: tuple[int, ...]) -> str:
  return "x".join(str(dim) for dim in shape)


def Percentile(values: list[float], percentile: float) -> float:
  if not values:
    return 0.0
  return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def Summarize(values_ms: list[float], batch_size: int) -> dict[str, float]:
  mean_ms = statistics.fmean(values_ms)
  return {
      "mean_ms": mean_ms,
      "median_ms": statistics.median(values_ms),
      "p90_ms": Percentile(values_ms, 90.0),
      "min_ms": min(values_ms),
      "max_ms": max(values_ms),
      "items_per_s": batch_size * 1000.0 / mean_ms,
  }


def FormatSummary(name: str, summary: dict[str, float]) -> str:
  return (
      f"{name:<10} mean={summary['mean_ms']:8.3f} ms  "
      f"median={summary['median_ms']:8.3f} ms  "
      f"p90={summary['p90_ms']:8.3f} ms  "
      f"min={summary['min_ms']:8.3f} ms  "
      f"max={summary['max_ms']:8.3f} ms  "
      f"throughput={summary['items_per_s']:8.2f} items/s"
  )


def CreateOrtSession(onnx_path: Path) -> ort.InferenceSession:
  available = ort.get_available_providers()
  if "CUDAExecutionProvider" not in available:
    raise RuntimeError(
        "CUDAExecutionProvider is unavailable. Check PATH for CUDA/cuDNN/ORT "
        "runtime DLLs before running this script."
    )

  session = ort.InferenceSession(
      str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
  )
  providers = session.get_providers()
  if not providers or providers[0] != "CUDAExecutionProvider":
    raise RuntimeError(f"ORT did not select CUDAExecutionProvider: {providers}")
  return session


def Synchronize(binding: ort.IOBinding) -> None:
  if hasattr(binding, "synchronize_inputs"):
    binding.synchronize_inputs()
  if hasattr(binding, "synchronize_outputs"):
    binding.synchronize_outputs()


def BenchmarkOrtCuda(
    session: ort.InferenceSession,
    shape: tuple[int, ...],
    warmup_runs: int,
    iterations: int,
) -> dict[str, float]:
  input_name = session.get_inputs()[0].name
  input_array = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
  input_value = ort.OrtValue.ortvalue_from_numpy(input_array, "cuda", 0)

  binding = session.io_binding()
  binding.bind_ortvalue_input(input_name, input_value)
  for output in session.get_outputs():
    binding.bind_output(output.name, "cuda", 0)

  for _ in range(warmup_runs):
    session.run_with_iobinding(binding)
    Synchronize(binding)

  latencies_ms: list[float] = []
  for _ in range(iterations):
    start = time.perf_counter()
    session.run_with_iobinding(binding)
    Synchronize(binding)
    latencies_ms.append((time.perf_counter() - start) * 1000.0)

  return Summarize(latencies_ms, shape[0])


def ParseTrtexecMetrics(output: str, batch_size: int) -> dict[str, float]:
  latency_match = re.search(
      r"Latency:\s*min = ([0-9.]+) ms, max = ([0-9.]+) ms, "
      r"mean = ([0-9.]+) ms, median = ([0-9.]+) ms, "
      r"percentile\(90%\) = ([0-9.]+) ms",
      output,
  )
  if latency_match is None:
    raise RuntimeError("Failed to parse trtexec latency metrics")

  gpu_match = re.search(
      r"GPU Compute Time:\s*min = [^,]+, max = [^,]+, mean = ([0-9.]+) ms",
      output,
  )

  min_ms = float(latency_match.group(1))
  max_ms = float(latency_match.group(2))
  mean_ms = float(latency_match.group(3))
  summary = {
      "mean_ms": mean_ms,
      "median_ms": float(latency_match.group(4)),
      "p90_ms": float(latency_match.group(5)),
      "min_ms": min_ms,
      "max_ms": max_ms,
      "items_per_s": batch_size * 1000.0 / mean_ms,
  }
  if gpu_match is not None:
    summary["gpu_compute_mean_ms"] = float(gpu_match.group(1))
  return summary


def BenchmarkTrt(
    trtexec_path: Path,
    engine_path: Path,
    input_name: str,
    shape: tuple[int, ...],
    warmup_ms: int,
    iterations: int,
) -> dict[str, float]:
  cmd = [
      str(trtexec_path),
      f"--loadEngine={engine_path}",
      f"--shapes={input_name}:{ShapeText(shape)}",
      f"--warmUp={warmup_ms}",
      "--duration=0",
      f"--iterations={iterations}",
      "--noDataTransfers",
  ]
  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    raise RuntimeError(
        "trtexec failed\n"
        f"command: {' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
  return ParseTrtexecMetrics(result.stdout + "\n" + result.stderr, shape[0])


def BuildParser() -> argparse.ArgumentParser:
  build_dir = DefaultBuildDir()
  parser = argparse.ArgumentParser(
      description="Compare ALIKED performance in TRT and ORT CUDA"
  )
  parser.add_argument(
      "--onnx",
      type=Path,
      default=Path(r"D:\Users\rick\Downloads\aliked.onnx"),
      help="Path to the unified ALIKED ONNX model",
  )
  parser.add_argument(
      "--engine",
      type=Path,
      default=Path(r"D:\Users\rick\Downloads\aliked.engine"),
      help="Path to the TensorRT engine",
  )
  parser.add_argument(
      "--trtexec",
      type=Path,
      default=Path(r"C:\Program Files\TensorRT-10.16.1.11\bin\trtexec.exe"),
      help="Path to trtexec.exe",
  )
  parser.add_argument(
      "--shape",
      default="1x3x1024x1024",
      help="Benchmark shape in NCHW format, e.g. 1x3x1024x1024",
  )
  parser.add_argument(
      "--warmup-runs",
      type=int,
      default=20,
      help="Warmup iterations for ORT CUDA",
  )
  parser.add_argument(
      "--iterations",
      type=int,
      default=80,
      help="Measured iterations for both backends",
  )
  parser.add_argument(
      "--trt-warmup-ms",
      type=int,
      default=1000,
      help="Warmup duration passed to trtexec in milliseconds",
  )
  return parser


def main() -> int:
  args = BuildParser().parse_args()
  shape = ParseShape(args.shape)
  if len(shape) != 4:
    raise ValueError("Expected a 4D NCHW shape")

  if not args.onnx.is_file():
    raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
  if not args.engine.is_file():
    raise FileNotFoundError(f"TensorRT engine not found: {args.engine}")
  if not args.trtexec.is_file():
    raise FileNotFoundError(f"trtexec not found: {args.trtexec}")

  session = CreateOrtSession(args.onnx)
  input_name = session.get_inputs()[0].name

  ort_summary = BenchmarkOrtCuda(
      session, shape, args.warmup_runs, args.iterations
  )
  trt_summary = BenchmarkTrt(
      args.trtexec,
      args.engine,
      input_name,
      shape,
      args.trt_warmup_ms,
      args.iterations,
  )

  speedup = ort_summary["mean_ms"] / trt_summary["mean_ms"]

  print(f"Model: {args.onnx}")
  print(f"Engine: {args.engine}")
  print(f"Shape: {ShapeText(shape)}")
  print(
      "Mode: GPU-resident IO only; excludes host-device transfers to reflect "
      "backend runtime cost"
  )
  print()
  print(FormatSummary("ORT CUDA", ort_summary))
  print(FormatSummary("TensorRT", trt_summary))
  if "gpu_compute_mean_ms" in trt_summary:
    print(
        f"TensorRT   GPU-compute-mean={trt_summary['gpu_compute_mean_ms']:8.3f} ms"
    )
  print(f"Speedup    TensorRT is {speedup:.2f}x faster than ORT CUDA")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())