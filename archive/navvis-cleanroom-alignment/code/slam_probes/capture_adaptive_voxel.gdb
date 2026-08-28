set pagination off
set confirm off
set disable-randomization on
starti
python
import gdb
import json
import os
import struct

IMAGE_BASE = 0x555555554000
ADAPTIVE_FILTER = IMAGE_BASE + 0x62e770
FIRST_POINT_FILTER = IMAGE_BASE + 0x638730
FIRST_POINT_KERNEL = IMAGE_BASE + 0x6381d0
DETERMINISTIC_INDEX_FILTER = IMAGE_BASE + 0x62d920

OUTPUT_DIR = os.environ.get("NAVVIS_ADAPTIVE_PROBE_DIR", "/tmp/navvis-adaptive-voxel")
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {
    "active": False,
    "adaptive_calls": [],
    "grid_calls": [],
    "index_calls": [],
}


def inferior():
    return gdb.selected_inferior()


def read(address, size):
    return bytes(inferior().read_memory(address, size))


def u32(address):
    return struct.unpack("<I", read(address, 4))[0]


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def f32(address):
    return struct.unpack("<f", read(address, 4))[0]


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
        stream.write(b"NVAVF24\0")
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
    }


def save_state():
    path = os.path.join(OUTPUT_DIR, "trace.json")
    with open(path, "w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)


class KernelFinish(gdb.FinishBreakpoint):
    def __init__(self, record, output_vector):
        super().__init__(internal=True)
        self.record = record
        self.output_vector = output_vector

    def stop(self):
        self.record["output_count"] = vector_count(self.output_vector)
        save_state()
        return False


class KernelBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % FIRST_POINT_KERNEL, internal=True)

    def stop(self):
        if not state["active"]:
            return False
        output_vector = int(gdb.parse_and_eval("$rdi"))
        input_vector = int(gdb.parse_and_eval("$rsi"))
        resolution = float(gdb.parse_and_eval("$xmm0.v4_float[0]"))
        record = {
            "call": len(state["grid_calls"]),
            "resolution_float32": resolution,
            "input_count": vector_count(input_vector),
            "output_count": None,
        }
        state["grid_calls"].append(record)
        KernelFinish(record, output_vector)
        return False


class IndexFinish(gdb.FinishBreakpoint):
    def __init__(self, record, output_vector):
        super().__init__(internal=True)
        self.record = record
        self.output_vector = output_vector

    def stop(self):
        self.record["output"] = dump_vector(
            "node0_index_output.bin", self.output_vector
        )
        save_state()
        return False


class IndexBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % DETERMINISTIC_INDEX_FILTER, internal=True)

    def stop(self):
        if not state["active"]:
            return False
        output_vector = int(gdb.parse_and_eval("$rdi"))
        input_vector = int(gdb.parse_and_eval("$rsi"))
        maximum_points = int(gdb.parse_and_eval("$rdx"))
        record = {
            "call": len(state["index_calls"]),
            "maximum_points": maximum_points,
            "input": dump_vector("node0_index_input.bin", input_vector),
            "output": None,
        }
        state["index_calls"].append(record)
        IndexFinish(record, output_vector)
        return False


class AdaptiveFinish(gdb.FinishBreakpoint):
    def __init__(self, record, output_vector):
        super().__init__(internal=True)
        self.record = record
        self.output_vector = output_vector

    def stop(self):
        self.record["output"] = dump_vector(
            "node0_adaptive_output.bin", self.output_vector
        )
        state["active"] = False
        save_state()
        gdb.write("captured node0 high-resolution adaptive filter\n")
        gdb.execute("quit")
        return False


class AdaptiveBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % ADAPTIVE_FILTER, internal=True)

    def stop(self):
        options = int(gdb.parse_and_eval("$rsi"))
        output_vector = int(gdb.parse_and_eval("$rdi"))
        input_vector = int(gdb.parse_and_eval("$rdx"))
        maximum_points = f32(options + 0x24)
        filter_type = u32(options + 0x20)
        input_count = vector_count(input_vector)
        if maximum_points != 5000.0 or input_count < 5000:
            return False
        state["active"] = True
        record = {
            "call": len(state["adaptive_calls"]),
            "options_address": options,
            "options_hex": read(options, 0x38).hex(),
            "min_or_max_length_at_0x18": f32(options + 0x18),
            "max_range_at_0x1c": f32(options + 0x1c),
            "voxel_filter_type": filter_type,
            "max_num_points": maximum_points,
            "other_length_at_0x28": f32(options + 0x28),
            "grid_filtering_enabled": bool(read(options + 0x2c, 1)[0]),
            "index_filtering_enabled": bool(read(options + 0x2d, 1)[0]),
            "index_filter_random": bool(read(options + 0x2e, 1)[0]),
            "max_num_iterations": u32(options + 0x30),
            "input": dump_vector("node0_adaptive_input.bin", input_vector),
            "output": None,
        }
        state["adaptive_calls"].append(record)
        save_state()
        gdb.write(
            "node0 adaptive input=%d type=%d lengths=(%.9g, %.9g) iterations=%d\n"
            % (
                input_count,
                filter_type,
                record["min_or_max_length_at_0x18"],
                record["other_length_at_0x28"],
                record["max_num_iterations"],
            )
        )
        AdaptiveFinish(record, output_vector)
        return False


AdaptiveBreakpoint()
KernelBreakpoint()
IndexBreakpoint()
end
continue
