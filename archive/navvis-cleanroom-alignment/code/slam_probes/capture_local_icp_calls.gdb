set pagination off
set confirm off
set breakpoint pending on
starti
python
import gdb
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT",
    "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_local_icp_calls_0_3",
))
MATCH_OFFSET = 0x6C4660
CORRESPONDENCE_OFFSET = 0x6CDC40
CALL_LIMIT = int(os.environ.get("NAVVIS_PROBE_CALL_LIMIT", "4"))


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


def memory(address, size):
    return bytes(gdb.selected_inferior().read_memory(address, size))


def dump_raw(path, address, size):
    data = memory(address, size)
    (OUTPUT / path).write_bytes(data)


def dump_vector(path, owner):
    begin, end, capacity = struct.unpack("<QQQ", memory(owner, 24))
    if not (begin <= end <= capacity):
        raise RuntimeError("invalid vector at %#x" % owner)
    dump_raw(path, begin, end - begin)
    return {"begin": begin, "end": end, "capacity": capacity,
            "used_bytes": end - begin, "path": path}


BASE = image_base("surveyorslam_processing_node")
OUTPUT.mkdir(parents=True, exist_ok=True)
state = {"calls": []}


def write_metadata():
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "executable_base": BASE,
        "match_offset": MATCH_OFFSET,
        "calls": state["calls"],
    }, indent=2, sort_keys=True))


class MatchReturn(gdb.FinishBreakpoint):
    def __init__(self, call_index, output_pointer):
        super().__init__(gdb.newest_frame(), internal=True)
        self.call_index = call_index
        self.output_pointer = output_pointer

    def stop(self):
        prefix = "call_%02d" % self.call_index
        dump_raw(prefix + "_result.bin", self.output_pointer, 64)
        state["calls"][self.call_index]["return_pc"] = int(gdb.parse_and_eval("$pc"))
        write_metadata()
        if self.call_index + 1 >= CALL_LIMIT:
            gdb.write("captured %d local ICP calls\n" % CALL_LIMIT)
            gdb.execute("quit")
        return False


class MatchBreakpoint(gdb.Breakpoint):
    def stop(self):
        call_index = len(state["calls"])
        if call_index >= CALL_LIMIT:
            return False
        registers = {
            name: int(gdb.parse_and_eval("$" + name))
            for name in ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
        }
        prefix = "call_%02d" % call_index
        record = {
            "index": call_index,
            "registers": registers,
            "source": dump_vector(prefix + "_source.bin", registers["rdx"]),
        }
        dump_raw(prefix + "_initial.bin", registers["rcx"], 64)
        dump_raw(prefix + "_normalization.bin", registers["r8"], 64)
        state["calls"].append(record)
        write_metadata()
        MatchReturn(call_index, registers["rdi"])
        return False


class CorrespondenceBreakpoint(gdb.Breakpoint):
    def stop(self):
        if not state["calls"]:
            return False
        record = state["calls"][-1]
        if "target_points" in record:
            return False
        call_index = record["index"]
        search = int(gdb.parse_and_eval("$rdi"))
        search_fields = struct.unpack("<QQQ", memory(search, 24))
        correspondence_owner = search_fields[0]
        solver = search_fields[1]
        point_owner, normal_owner = struct.unpack("<QQ", memory(solver + 8, 16))
        prefix = "call_%02d" % call_index
        record["search"] = search
        record["solver"] = solver
        finder_begin, finder_end, _ = struct.unpack(
            "<QQQ", memory(correspondence_owner + 8, 24)
        )
        record["target_levels"] = []
        for level, finder_entry in enumerate(range(finder_begin, finder_end, 16)):
            finder = struct.unpack("<Q", memory(finder_entry, 8))[0]
            point_vector_owner = struct.unpack("<Q", memory(finder + 0x38, 8))[0]
            record["target_levels"].append(dump_vector(
                prefix + "_target_level_%d.bin" % level,
                point_vector_owner,
            ))
        record["target_points"] = dump_vector(
            prefix + "_target_points.bin", point_owner
        )
        record["target_normals"] = dump_vector(
            prefix + "_target_normals.bin", normal_owner
        )
        write_metadata()
        return False


MatchBreakpoint("*%#x" % (BASE + MATCH_OFFSET), internal=True)
CorrespondenceBreakpoint("*%#x" % (BASE + CORRESPONDENCE_OFFSET), internal=True)
gdb.write("capturing the first %d local ICP calls at base %#x\n" % (CALL_LIMIT, BASE))
end
continue
