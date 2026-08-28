set pagination off
set confirm off
starti
python
import gdb
import json
import struct

IMAGE_BASE = 0x555555554000
PROCESS = IMAGE_BASE + 0x3bc5f0
PREDICATE = IMAGE_BASE + 0x3b7ad0
OUTPUT = "/tmp/navvis_vendor_minimum_range_predicates.json"

records = []

def memory(address, size):
    if not address:
        return None

def u64(address):
    return struct.unpack("<Q", bytes(gdb.selected_inferior().read_memory(address, 8)))[0]

def u32(address):
    return struct.unpack("<I", bytes(gdb.selected_inferior().read_memory(address, 4)))[0]

def floats(address, count):
    return list(struct.unpack(
        "<%df" % count,
        bytes(gdb.selected_inferior().read_memory(address, 4 * count)),
    ))

VTABLE_OFFSETS = {
    0x4b6a8: "sphere",
    0x4b6d0: "cylindrical",
    0x4b6f8: "transformed",
    0x4b720: "box",
    0x4b748: "boolean",
    0x4b770: "yaw_range",
    0x4b798: "pitch_range",
}

def serialize_region(pointer, library_base, seen, depth=0):
    if not pointer or depth > 16 or pointer in seen:
        return None
    seen.add(pointer)
    vptr = u64(pointer)
    kind = VTABLE_OFFSETS.get(vptr - library_base, "unknown")
    result = {"pointer": pointer, "vptr": vptr, "kind": kind}
    if kind in ("sphere", "cylindrical", "yaw_range", "pitch_range"):
        result["parameters"] = floats(pointer + 8, 2)
    elif kind == "box":
        result["minimum"] = floats(pointer + 8, 3)
        result["maximum"] = floats(pointer + 0x14, 3)
    elif kind == "transformed":
        child = u64(pointer + 8)
        result["matrix_column_major"] = floats(pointer + 0x20, 16)
        result["child"] = serialize_region(child, library_base, seen, depth + 1)
    elif kind == "boolean":
        begin = u64(pointer + 8)
        end = u64(pointer + 0x10)
        result["entries"] = []
        if end >= begin and (end - begin) % 24 == 0 and end - begin < 24000:
            for entry in range(begin, end, 24):
                child = u64(entry + 8)
                result["entries"].append({
                    "operation": u32(entry),
                    "child": serialize_region(child, library_base, seen, depth + 1),
                })
    return result
    try:
        return bytes(gdb.selected_inferior().read_memory(address, size)).hex()
    except gdb.MemoryError:
        return None

class PredicateBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % PREDICATE, internal=True)
        self.enabled = False
        self.process_hit = 0

    def stop(self):
        rule = int(gdb.parse_and_eval("$rdi"))
        point = int(gdb.parse_and_eval("$rsi"))
        transform = int(gdb.parse_and_eval("$rdx"))
        frame_case = int(gdb.parse_and_eval("$ecx"))
        vptr = u64(rule)
        # The second minimum-range rule is a libnavvis RegionTransformed.
        is_local_sphere = (vptr - IMAGE_BASE) == 0x1825740
        library_base = vptr - 0x4b6f8
        records.append({
            "process_hit": self.process_hit,
            "rule": rule,
            "rule_hex": memory(rule, 0xa0),
            "rule_child": (
                int(gdb.parse_and_eval("*(unsigned long long*)%d" % (rule + 8)))
                if rule else 0
            ),
            "point": point,
            "point_hex": memory(point, 0x20),
            "transform": transform,
            "transform_hex": memory(transform, 0x40),
            "frame_case": frame_case,
            "rule_tree": (
                {"kind": "local_sphere", "parameters": floats(rule + 8, 2)}
                if is_local_sphere
                else serialize_region(rule, library_base, set())
            ),
        })
        records[-1]["rule_child_hex"] = memory(records[-1]["rule_child"], 0x80)
        with open(OUTPUT, "w") as stream:
            json.dump(records, stream, indent=2, sort_keys=True)
        self.enabled = False
        return False

predicate_breakpoint = PredicateBreakpoint()

class ProcessBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % PROCESS, internal=True)
        self.hits = 0

    def stop(self):
        self.hits += 1
        if self.hits <= 8:
            predicate_breakpoint.process_hit = self.hits
            predicate_breakpoint.enabled = True
        return False

ProcessBreakpoint()
end
continue
