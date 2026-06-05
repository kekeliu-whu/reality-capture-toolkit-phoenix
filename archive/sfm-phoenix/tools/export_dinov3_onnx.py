"""
Export DINOv3 retrieval model to ONNX for Phoenix.

Phoenix retrieval expects:
  input:  pixel_values [B, 3, 224, 224]
  output: embeddings   [B, 768]

Example (ModelScope cache first, download if missing):
  python sfm-phoenix/tools/export_dinov3_onnx.py \
    --download-from-modelscope \
    --modelscope-model-id facebook/dinov3-vitb16-pretrain-lvd1689m \
    --output sfm-phoenix/models/dinov3_vitb16_pretrain_lvd1689m.onnx \
    --export-fp16
"""

import argparse
import os
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from transformers import AutoModel


class DinoV3RetrievalWrapper(nn.Module):
  """Wrap DINOv3 to expose a single embedding tensor output."""

  def __init__(self, backbone: nn.Module):
    super().__init__()
    self.backbone = backbone

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = self.backbone(pixel_values=pixel_values)
    return outputs.pooler_output


def GuessModelScopeCachePath(model_id: str, cache_dir: str) -> Path:
  if cache_dir:
    base_dir = Path(cache_dir)
  else:
    base_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models"
  return base_dir / model_id.replace("/", os.sep)


def DownloadFromModelScope(model_id: str,
                           cache_dir: str,
                           revision: str) -> str:
  try:
    from modelscope.hub import snapshot_download as snapshot_download_entry
  except ImportError as exc:
    raise RuntimeError(
        "modelscope is required for downloading. "
        "Install with: pip install modelscope") from exc

  if callable(snapshot_download_entry):
    snapshot_download_fn = snapshot_download_entry
  else:
    snapshot_download_fn = snapshot_download_entry.snapshot_download

  kwargs = {"model_id": model_id}
  if cache_dir:
    kwargs["cache_dir"] = cache_dir
  if revision:
    kwargs["revision"] = revision

  model_dir = snapshot_download_fn(**kwargs)
  if not model_dir:
    raise RuntimeError(f"Failed to download model from ModelScope: {model_id}")
  return model_dir


def ResolveModelDirectory(model_dir: str,
                          download_from_modelscope: bool,
                          modelscope_model_id: str,
                          modelscope_cache_dir: str,
                          modelscope_revision: str) -> str:
  if model_dir:
    resolved = Path(model_dir).expanduser()
    if resolved.is_dir():
      print(f"Using local model directory: {resolved}")
      return str(resolved)
    if resolved.exists():
      raise NotADirectoryError(f"Model path is not a directory: {resolved}")
    if not download_from_modelscope:
      raise FileNotFoundError(
          "Model directory not found and ModelScope download is disabled: "
          f"{resolved}")

  cached_dir = GuessModelScopeCachePath(modelscope_model_id,
                                        modelscope_cache_dir)
  if cached_dir.is_dir():
    print(f"Using existing ModelScope cache: {cached_dir}")
    return str(cached_dir)

  if not download_from_modelscope:
    raise FileNotFoundError(
        "ModelScope cache not found and download disabled. "
        f"Missing: {cached_dir}")

  print("ModelScope cache not found. Downloading model: "
        f"{modelscope_model_id}")
  downloaded_dir = DownloadFromModelScope(
      model_id=modelscope_model_id,
      cache_dir=modelscope_cache_dir,
      revision=modelscope_revision)
  print(f"Downloaded model to: {downloaded_dir}")
  return downloaded_dir


def ResolveDevice(device_arg: str) -> str:
  if device_arg == "cuda" and not torch.cuda.is_available():
    print("CUDA is unavailable, falling back to CPU for export.")
    return "cpu"
  return device_arg


def ExportDinov3Onnx(model_dir: str,
                     output_path: str,
                     device: str,
                     opset: int,
                     dummy_batch: int,
                     local_files_only: bool,
                     export_fp16: bool) -> None:
  if export_fp16 and device != "cuda":
    raise RuntimeError("FP16 ONNX export requires CUDA device")

  export_dtype = torch.float16 if export_fp16 else torch.float32

  print(f"Loading DINOv3 from: {model_dir}")
  backbone = AutoModel.from_pretrained(
      model_dir,
      local_files_only=local_files_only,
  )
  backbone.eval()
  if export_fp16:
    backbone.half()
  backbone.to(device)

  model = DinoV3RetrievalWrapper(backbone).eval().to(device)
  if export_fp16:
    model.half()

  dummy = torch.randn(dummy_batch,
                      3,
                      224,
                      224,
                      device=device,
                      dtype=export_dtype)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  print(f"Exporting ONNX in {'FP16' if export_fp16 else 'FP32'}...")
  torch.onnx.export(
      model,
      (dummy,),
      output_path,
      opset_version=opset,
      input_names=["pixel_values"],
      output_names=["embeddings"],
      dynamic_axes={
          "pixel_values": {0: "batch"},
          "embeddings": {0: "batch"},
      },
      dynamo=False,
  )

  onnx_model = onnx.load(output_path)
  onnx.checker.check_model(onnx_model)
  onnx.save(onnx_model, output_path)

  print(f"Saved: {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Export DINOv3 to ONNX")
  parser.add_argument(
      "--model-dir",
    default="",
    help="Local model directory. If empty, resolve/download from ModelScope",
  )
  parser.add_argument(
    "--modelscope-model-id",
    default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    help="ModelScope model id",
  )
  parser.add_argument(
    "--modelscope-cache-dir",
    default="",
    help="Optional ModelScope cache directory",
  )
  parser.add_argument(
    "--modelscope-revision",
    default="",
    help="Optional ModelScope revision",
  )
  parser.add_argument(
    "--download-from-modelscope",
    dest="download_from_modelscope",
    action="store_true",
    help="Download from ModelScope if cache/model-dir is missing",
  )
  parser.add_argument(
    "--no-modelscope-download",
    dest="download_from_modelscope",
    action="store_false",
    help="Disable ModelScope download",
  )
  parser.add_argument(
      "--output",
      default="sfm-phoenix/models/dinov3_vitb16_pretrain_lvd1689m.onnx",
      help="Output ONNX path",
  )
  parser.add_argument(
      "--device",
      default="cuda",
      choices=["cuda", "cpu"],
      help="Device used during export",
  )
  parser.add_argument(
      "--opset",
      type=int,
      default=19,
      help="ONNX opset version",
  )
  parser.add_argument(
      "--dummy-batch",
      type=int,
      default=2,
      help="Dummy batch size used at export time",
  )
  parser.add_argument(
      "--local-files-only",
      dest="local_files_only",
      action="store_true",
      help="Disable online fetching and load model from local cache only",
  )
  parser.add_argument(
      "--allow-online-model-load",
      dest="local_files_only",
      action="store_false",
      help="Allow transformers to fetch remote files during model loading",
  )
  parser.add_argument(
      "--export-fp16",
      dest="export_fp16",
      action="store_true",
      help="Export ONNX weights and graph in FP16",
  )
  parser.add_argument(
      "--export-fp32",
      dest="export_fp16",
      action="store_false",
      help="Export ONNX in FP32",
  )
  parser.set_defaults(download_from_modelscope=True)
  parser.set_defaults(local_files_only=True)
  parser.set_defaults(export_fp16=True)

  args = parser.parse_args()

  resolved_model_dir = ResolveModelDirectory(
      model_dir=args.model_dir,
      download_from_modelscope=args.download_from_modelscope,
      modelscope_model_id=args.modelscope_model_id,
      modelscope_cache_dir=args.modelscope_cache_dir,
      modelscope_revision=args.modelscope_revision,
  )

  device = ResolveDevice(args.device)
  ExportDinov3Onnx(
      model_dir=resolved_model_dir,
      output_path=args.output,
      device=device,
      opset=args.opset,
      dummy_batch=args.dummy_batch,
      local_files_only=args.local_files_only,
      export_fp16=args.export_fp16,
  )


if __name__ == "__main__":
  main()
