set pagination off
set confirm off
set disable-randomization on
starti
python
import gdb
import hashlib
import json
import os
import struct

IMAGE_BASE = 0x555555554000
RANGE_ADAPTIVE_WRAPPER = IMAGE_BASE + 0x62ef30
ADAPTIVE_FILTER = IMAGE_BASE + 0x62e770
RAW_RANGE_LOOP = IMAGE_BASE + 0x4a7b5c
CENTROID_KERNEL = IMAGE_BASE + 0x636050

OUTPUT_DIR = os.environ.get(
    "NAVVIS_HYBRID_RANGE_PROBE_DIR", "/tmp/navvis-hybrid-range-boundary"
)
TARGET_NODE = int(os.environ.get("NAVVIS_HYBRID_NODE_INDEX", "0"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {
    "active": False,
    "raw_boundary_captured": False,
    "centroid_captured": False,
    "wrapper_calls_seen": 0,
    "qualifying_wrapper_calls": 0,
    "qualifying_raw_calls": 0,
    "qualifying_centroid_calls": 0,
    "target_node": TARGET_NODE,
    "capture": None,
}


def inferior():
    return gdb.selected_inferior()


def read(address, size):
    return bytes(inferior().read_memory(address, size))


def f32(address):
    return struct.unpack("<f", read(address, 4))[0]


def u32(address):
    return struct.unpack("<I", read(address, 4))[0]


def vector_bounds(vector):
    begin, end, capacity = struct.unpack("<QQQ", read(vector, 24))
    if end < begin or capacity < end or (end - begin) % 24:
        raise RuntimeError(
            "invalid 24-byte vector at %#x: %#x %#x %#x"
            % (vector, begin, end, capacity)
        )
    return begin, end, capacity


def vector_count(vector):
    begin, end, _ = vector_bounds(vector)
    return (end - begin) // 24


def dump_vector(name, vector):
    begin, end, capacity = vector_bounds(vector)
    payload = read(begin, end - begin) if end != begin else b""
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "wb") as stream:
        stream.write(b"NVHRF24\0")
        stream.write(struct.pack("<QQQQ", vector, begin, end, capacity))
        stream.write(payload)
    return {
        "path": path,
        "vector": vector,
        "begin": begin,
        "end": end,
        "capacity": capacity,
        "count": (end - begin) // 24,
        "record_size": 24,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def save_state():
    with open(os.path.join(OUTPUT_DIR, "trace.json"), "w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)


class WrapperBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % RANGE_ADAPTIVE_WRAPPER, internal=True)

    def stop(self):
        state["wrapper_calls_seen"] += 1
        options = int(gdb.parse_and_eval("$rsi"))
        input_vector = int(gdb.parse_and_eval("$rdx"))
        count = vector_count(input_vector)
        maximum_points = f32(options + 0x24)
        if maximum_points != 5000.0 or count < 5000:
            return False
        candidate = state["qualifying_wrapper_calls"]
        state["qualifying_wrapper_calls"] += 1
        if state["active"] or candidate != TARGET_NODE:
            return False
        state["active"] = True
        state["capture"] = {
            "raw_range_loop_rva": "0x4a7b5c",
            "wrapper_rva": "0x62ef30",
            "adaptive_rva": "0x62e770",
            "options_address": options,
            "options_hex": read(options, 0x38).hex(),
            "max_length": f32(options + 0x18),
            "max_range": f32(options + 0x1c),
            "voxel_filter_type": u32(options + 0x20),
            "max_num_points": maximum_points,
            "min_length": f32(options + 0x28),
            "grid_filtering_enabled": bool(read(options + 0x2c, 1)[0]),
            "index_filtering_enabled": bool(read(options + 0x2d, 1)[0]),
            "index_filter_random": bool(read(options + 0x2e, 1)[0]),
            "max_num_iterations": u32(options + 0x30),
            "wrapper_input": dump_vector("node0_wrapper_input.bin", input_vector),
            "adaptive_input": None,
        }
        save_state()
        gdb.write("captured candidate wrapper input count=%d\n" % count)
        return False


class RawRangeLoopBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % RAW_RANGE_LOOP, internal=True)

    def stop(self):
        if state["raw_boundary_captured"]:
            return False
        vector = int(gdb.parse_and_eval("$rsp")) + 0x40
        count = vector_count(vector)
        if count < 5000:
            return False
        candidate = state["qualifying_raw_calls"]
        state["qualifying_raw_calls"] += 1
        if candidate != TARGET_NODE:
            return False
        options = int(gdb.parse_and_eval("$rbp"))
        state["raw_boundary_captured"] = True
        state["raw_range_loop"] = {
            "rva": "0x4a7b5c",
            "input": dump_vector("node0_raw_range_input.bin", vector),
            "range_squared_lower_at_0x348": f32(options + 0x348),
            "range_squared_upper_at_0x34c": f32(options + 0x34c),
            "clipped_ray_length_at_0x94": f32(options + 0x94),
            "transform_float32_hex": read(
                int(gdb.parse_and_eval("$rsp")) + 0x1e0, 64
            ).hex(),
        }
        save_state()
        gdb.write("captured node0 raw range-loop input count=%d\n" % count)
        return False


class CentroidFinish(gdb.FinishBreakpoint):
    def __init__(self, output_vector):
        super().__init__(internal=True)
        self.output_vector = output_vector

    def stop(self):
        state["centroid_filter"]["output"] = dump_vector(
            "node0_centroid_output.bin", self.output_vector
        )
        state["centroid_captured"] = True
        save_state()
        gdb.write(
            "captured node0 centroid boundary %d -> %d\n"
            % (
                state["centroid_filter"]["input"]["count"],
                state["centroid_filter"]["output"]["count"],
            )
        )
        return False


class CentroidBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % CENTROID_KERNEL, internal=True)

    def stop(self):
        if state["centroid_captured"] or "centroid_filter" in state:
            return False
        output_vector = int(gdb.parse_and_eval("$rdi"))
        input_vector = int(gdb.parse_and_eval("$rsi"))
        count = vector_count(input_vector)
        resolution = float(gdb.parse_and_eval("$xmm0.v4_float[0]"))
        if count < 5000 or abs(resolution - 0.04) > 1e-6:
            return False
        candidate = state["qualifying_centroid_calls"]
        state["qualifying_centroid_calls"] += 1
        if candidate != TARGET_NODE:
            return False
        state["centroid_filter"] = {
            "kernel_rva": "0x636050",
            "resolution_float32": resolution,
            "input": dump_vector("node0_centroid_input.bin", input_vector),
            "output": None,
        }
        save_state()
        CentroidFinish(output_vector)
        return False


class AdaptiveBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % ADAPTIVE_FILTER, internal=True)

    def stop(self):
        if not state["active"]:
            return False
        options = int(gdb.parse_and_eval("$rsi"))
        if options != state["capture"]["options_address"]:
            return False
        input_vector = int(gdb.parse_and_eval("$rdx"))
        state["capture"]["adaptive_input"] = dump_vector(
            "node0_adaptive_input.bin", input_vector
        )
        save_state()
        gdb.write(
            "captured node0 wrapper boundary %d -> %d\n"
            % (
                state["capture"]["wrapper_input"]["count"],
                state["capture"]["adaptive_input"]["count"],
            )
        )
        gdb.execute("quit")
        return False


WrapperBreakpoint()
AdaptiveBreakpoint()
RawRangeLoopBreakpoint()
CentroidBreakpoint()
end
continue
