set pagination off
set confirm off
starti
python
import gdb
import json
import struct

IMAGE_BASE = 0x555555554000
PROCESS = IMAGE_BASE + 0x3b9b20
OUTPUT = "/tmp/navvis_vendor_filter_region_trees.json"

VTABLE_OFFSETS = {
    0x4b6a8: "sphere",
    0x4b6d0: "cylindrical",
    0x4b6f8: "transformed",
    0x4b720: "box",
    0x4b748: "boolean",
    0x4b770: "yaw_range",
    0x4b798: "pitch_range",
}

def read(address, size):
    return bytes(gdb.selected_inferior().read_memory(address, size))

def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]

def u32(address):
    return struct.unpack("<I", read(address, 4))[0]

def floats(address, count):
    return list(struct.unpack("<%df" % count, read(address, 4 * count)))

def serialize_region(pointer, library_base, seen, depth=0):
    if not pointer:
        return None
    if depth > 16:
        return {"pointer": pointer, "error": "depth"}
    if pointer in seen:
        return {"pointer": pointer, "reference": True}
    seen.add(pointer)
    vptr = u64(pointer)
    offset = vptr - library_base
    kind = VTABLE_OFFSETS.get(offset, "unknown")
    result = {
        "pointer": pointer,
        "vptr": vptr,
        "vptr_offset": offset,
        "kind": kind,
        "raw_hex": read(pointer, 0x70).hex(),
    }
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
        result["begin"] = begin
        result["end"] = end
        result["entries"] = []
        if end >= begin and (end - begin) % 24 == 0 and end - begin <= 24 * 1000:
            for entry in range(begin, end, 24):
                child = u64(entry + 8)
                result["entries"].append({
                    "operation": u32(entry),
                    "child_pointer": child,
                    "child": serialize_region(child, library_base, seen, depth + 1),
                })
        else:
            result["error"] = "invalid boolean vector"
    return result

class RegionTreeBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % PROCESS, internal=True)
        self.seen = set()
        self.records = []

    def stop(self):
        owner = int(gdb.parse_and_eval("$rdi"))
        if owner in self.seen:
            return False
        self.seen.add(owner)
        root = u64(owner + 0x88)
        root_vptr = u64(root)
        library_base = root_vptr - 0x4b748
        record = {
            "owner": owner,
            "owner_hex": read(owner, 0x98).hex(),
            "intensity_threshold": floats(owner + 0x80, 1)[0],
            "region_pointer": root,
            "library_base": library_base,
            "region": serialize_region(root, library_base, set()),
        }
        self.records.append(record)
        with open(OUTPUT, "w") as stream:
            json.dump(self.records, stream, indent=2, sort_keys=True)
        gdb.write("captured filter region tree %d owner=%#x threshold=%g\n" %
                  (len(self.records), owner, record["intensity_threshold"]))
        return False

RegionTreeBreakpoint()
end
continue
