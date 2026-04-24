"""
Export DINOv3 retrieval model to ONNX for Phoenix.

Phoenix retrieval expects:
  input:  pixel_values [B, 3, 224, 224] float32
  output: embeddings   [B, 768] float32

Example:
  python sfm-phoenix/tools/export_dinov3_onnx.py \
    --model-dir "C:/Users/rick/.cache/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --output "sfm-phoenix/models/dinov3_vitb16_pretrain_lvd1689m.onnx"
"""

import argparse
import os

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
                     local_files_only: bool) -> None:
  print(f"Loading DINOv3 from: {model_dir}")
  backbone = AutoModel.from_pretrained(
      model_dir,
      local_files_only=local_files_only,
  )
  backbone.eval().to(device)

  model = DinoV3RetrievalWrapper(backbone).eval().to(device)

  dummy = torch.randn(dummy_batch, 3, 224, 224, device=device)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  print("Exporting ONNX...")
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
      default="facebook/dinov3-vitb16-pretrain-lvd1689m",
      help="Local directory or HuggingFace model id",
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
      action="store_true",
      help="Disable online fetching and load model from local cache only",
  )
  args = parser.parse_args()

  device = ResolveDevice(args.device)
  ExportDinov3Onnx(
      model_dir=args.model_dir,
      output_path=args.output,
      device=device,
      opset=args.opset,
      dummy_batch=args.dummy_batch,
      local_files_only=args.local_files_only,
  )


if __name__ == "__main__":
  main()
