"""
Convert ONNX models to TensorRT engines with FP16 optimisation.

Usage:
    python feature_extraction/tools/build_trt_engines.py \
        --input feature_extraction/models/ \
        --output feature_extraction/engines/
"""

import os
import sys
import argparse

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt not installed.  pip install tensorrt")
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: float = 4.0,
    profiles: list = None,
):
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path:   Path to the ONNX model.
        engine_path: Output path for the serialised TRT engine.
        fp16:        Enable FP16 precision.
        workspace_gb: Max workspace size in GB.
        profiles:    List of (name, min_shape, opt_shape, max_shape) tuples
                     for dynamic axes.  Each shape is a tuple of ints.
    """
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  FP16 enabled")

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")

    # Set up dynamic shape optimization profiles
    if profiles:
        profile = builder.create_optimization_profile()
        for name, min_s, opt_s, max_s in profiles:
            profile.set_shape(name, min_s, opt_s, max_s)
        config.add_optimization_profile(profile)

    print(f"  Building engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f"  Saved: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


def build_aliked_backbone(input_dir: str, output_dir: str, fp16: bool):
    onnx = os.path.join(input_dir, 'aliked_backbone.onnx')
    engine = os.path.join(output_dir, 'aliked_backbone.engine')
    if not os.path.exists(onnx):
        print(f"  Skipping backbone: {onnx} not found")
        return
    print(f"\n[aliked_backbone]")
    build_engine(onnx, engine, fp16=fp16, profiles=[
        # (input_name, min_shape, opt_shape, max_shape)
        ('image', (1, 3, 320, 320), (1, 3, 640, 960), (1, 3, 1024, 1920)),
    ])


def build_aliked_sddh(input_dir: str, output_dir: str, fp16: bool):
    onnx = os.path.join(input_dir, 'aliked_sddh.onnx')
    engine = os.path.join(output_dir, 'aliked_sddh.engine')
    if not os.path.exists(onnx):
        print(f"  Skipping SDDH: {onnx} not found")
        return
    print(f"\n[aliked_sddh]")
    build_engine(onnx, engine, fp16=fp16, profiles=[
        ('feature_map',  (1, 128, 320, 320), (1, 128, 640, 960), (1, 128, 1024, 1920)),
        ('keypoints_wh', (100, 2),           (2000, 2),          (5000, 2)),
    ])


def build_lightglue(input_dir: str, output_dir: str, fp16: bool):
    onnx = os.path.join(input_dir, 'lightglue.onnx')
    engine = os.path.join(output_dir, 'lightglue.engine')
    if not os.path.exists(onnx):
        print(f"  Skipping LightGlue: {onnx} not found")
        return
    print(f"\n[lightglue]")
    build_engine(onnx, engine, fp16=fp16, profiles=[
        ('kpts0', (1, 100, 2),  (1, 2000, 2),  (1, 5000, 2)),
        ('desc0', (1, 100, 128),(1, 2000, 128), (1, 5000, 128)),
        ('kpts1', (1, 100, 2),  (1, 2000, 2),  (1, 5000, 2)),
        ('desc1', (1, 100, 128),(1, 2000, 128), (1, 5000, 128)),
    ])


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engines from ONNX')
    parser.add_argument('--input', default='feature_extraction/models/',
                        help='Directory containing ONNX models')
    parser.add_argument('--output', default='feature_extraction/engines/',
                        help='Output directory for TRT engines')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    fp16 = not args.no_fp16

    build_aliked_backbone(args.input, args.output, fp16)
    build_aliked_sddh(args.input, args.output, fp16)
    build_lightglue(args.input, args.output, fp16)

    print("\nAll engines built. Ready for C++ inference.")


if __name__ == '__main__':
    main()
