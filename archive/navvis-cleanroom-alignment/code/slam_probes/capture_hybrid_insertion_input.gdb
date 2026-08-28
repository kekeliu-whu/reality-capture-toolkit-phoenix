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
# Submaps::InsertRangeData has transformed and range-clipped the RayData at
# this instruction.  rsp+0x30 is the std::vector<geometry::Ray<float, 3>>
# passed immediately afterwards to RangeDataInserter3D.
HYBRID_INPUT_READY = IMAGE_BASE + 0x603498
OUTPUT_DIR = os.environ.get(
    "NAVVIS_HYBRID_INSERTION_PROBE_DIR", "/tmp/navvis-hybrid-insertion-input"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read(address, size):
    return bytes(gdb.selected_inferior().read_memory(address, size))


class HybridInputBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % HYBRID_INPUT_READY, internal=True)

    def stop(self):
        stack = int(gdb.parse_and_eval("$rsp"))
        vector = stack + 0x30
        begin, end, capacity = struct.unpack("<QQQ", read(vector, 24))
        if end < begin or capacity < end or (end - begin) % 24:
            raise RuntimeError("invalid Ray vector at %#x" % vector)
        payload = read(begin, end - begin) if end != begin else b""
        with open(os.path.join(OUTPUT_DIR, "hybrid_rays.bin"), "wb") as stream:
            stream.write(payload)
        result = {
            "breakpoint_rva": "0x603498",
            "vector_address": vector,
            "begin": begin,
            "end": end,
            "capacity": capacity,
            "record_size": 24,
            "count": (end - begin) // 24,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "inserter_context_prefix_hex": read(
                int(gdb.parse_and_eval("$rbp")), 0x40
            ).hex(),
        }
        with open(os.path.join(OUTPUT_DIR, "trace.json"), "w") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        gdb.write("captured HybridGrid insertion input count=%d\n" % result["count"])
        gdb.execute("quit")
        return False


HybridInputBreakpoint()
end
continue
