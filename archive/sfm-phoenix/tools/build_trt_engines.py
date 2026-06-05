"""
Convert ONNX models to TensorRT engines with FP16 optimisation.

Usage:
    # Build ALIKED and LightGlue together (default)
    python sfm-phoenix/tools/build_trt_engines.py \
        --aliked D:\\Users\\rick\\Downloads\\aliked.onnx \
        --lightglue sfm-phoenix/models/lightglue.onnx \
        --output sfm-phoenix/build/RelWithDebInfo/

    # Build only LightGlue
    python sfm-phoenix/tools/build_trt_engines.py \
        --only lightglue \
        --lightglue sfm-phoenix/models/lightglue.onnx \
        --output sfm-phoenix/build/RelWithDebInfo/
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
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


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

    # Use maximum optimization level for best kernel selection (0-5, default 3)
    if hasattr(builder, 'builder_optimization_level'):
        builder.builder_optimization_level = 5
        print(f"  Builder optimization level: 5 (maximum)")

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  FP16 enabled")

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Use parse_from_file so TRT resolves external data relative to the ONNX file
    onnx_abs = os.path.abspath(onnx_path)
    if not parser.parse_from_file(onnx_abs):
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


def build_aliked(onnx_path: str, output_dir: str, fp16: bool):
    """Build TRT engine for the unified ALIKED ONNX model.

    Expected ONNX I/O (single image, batch=1):
        input:  image         [1, 3, H, W]  float32
        output: keypoints     [N, 2]        float32
                descriptors   [N, 128]      float32
                scores        [N]           float32
    """
    if not os.path.exists(onnx_path):
        print(f"  Skipping ALIKED: {onnx_path} not found")
        return
    engine_path = os.path.splitext(onnx_path)[0] + '.engine'
    print(f"\n[aliked]  {onnx_path}")
    build_engine(onnx_path, engine_path, fp16=fp16, profiles=[
        ('image', (1, 3, 320, 320), (1, 3, 1024, 1024), (1, 3, 1600, 1600)),
    ])


def build_lightglue(input_dir: str, output_dir: str, fp16: bool):
    onnx = os.path.join(input_dir, 'lightglue.onnx')
    engine = os.path.join(output_dir, 'lightglue_b4.engine')
    if not os.path.exists(onnx):
        print(f"  Skipping LightGlue: {onnx} not found")
        return
    print(f"\n[lightglue]  {onnx}")
    build_engine(onnx, engine, fp16=fp16, profiles=[
        # Phoenix feature matching supports batch sizes {1, 4, 8}.
        ('kpts0', (1, 100, 2),   (4, 5000, 2),   (8, 5000, 2)),
        ('desc0', (1, 100, 128), (4, 5000, 128), (8, 5000, 128)),
        ('kpts1', (1, 100, 2),   (4, 5000, 2),   (8, 5000, 2)),
        ('desc1', (1, 100, 128), (4, 5000, 128), (8, 5000, 128)),
    ])


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engines from ONNX')
    parser.add_argument('--aliked',
                        default=r'D:\Users\rick\Downloads\aliked.onnx',
                        help='Path to the unified ALIKED ONNX model')
    parser.add_argument('--lightglue', default=None,
                        help='Path to the LightGlue ONNX model '
                             '(or the directory containing lightglue.onnx)')
    parser.add_argument('--output', default=None,
                        help='Output directory for TRT engines; '
                             'defaults to the same directory as the source ONNX')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 for all engines')
    parser.add_argument('--only', choices=['aliked', 'lightglue'],
                        help='Build only the specified engine')
    args = parser.parse_args()

    fp16 = not args.no_fp16

    # Resolve LightGlue ONNX path
    lg_onnx = None
    if args.lightglue:
        if os.path.isdir(args.lightglue):
            lg_onnx = os.path.join(args.lightglue, 'lightglue.onnx')
        else:
            lg_onnx = args.lightglue

    output_dir = args.output  # may be None → each build func derives the path

    if not args.only or args.only == 'aliked':
        build_aliked(args.aliked, output_dir, fp16)
    if not args.only or args.only == 'lightglue':
        if lg_onnx is None:
            print("Skipping LightGlue: no --lightglue path given")
        else:
            # Derive output dir from LightGlue ONNX if not specified
            lg_out = output_dir or os.path.dirname(lg_onnx)
            build_lightglue(os.path.dirname(lg_onnx), lg_out, fp16)

    print("\nAll engines built. Ready for C++ inference.")


if __name__ == '__main__':
    main()
