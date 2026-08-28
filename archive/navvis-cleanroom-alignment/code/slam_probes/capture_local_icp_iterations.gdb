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
    "NAVVIS_LOCAL_ICP_ITERATION_DIR", "/tmp/navvis-local-icp-iterations"
))
TARGET_MATCH_INDEX = int(os.environ.get("NAVVIS_LOCAL_ICP_MATCH_INDEX", "0"))
MATCH_OFFSET = 0x6C4660
CORRESPONDENCE_OFFSET = 0x6CDC40
CORRESPONDENCE_RETURN_OFFSET = 0x6CEB6B


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


def dump_raw(name, address, size):
    try:
        data = memory(address, size)
    except gdb.MemoryError:
        return False
    (OUTPUT / name).write_bytes(data)
    return True


def read_vector(address):
    begin, end, capacity = struct.unpack("<QQQ", memory(address, 24))
    if not (begin <= end <= capacity):
        raise RuntimeError("invalid std::vector at %#x" % address)
    return begin, end, capacity


def dump_vector(name, address, maximum_bytes=64_000_000):
    begin, end, capacity = read_vector(address)
    used = end - begin
    record = {
        "address": address,
        "begin": begin,
        "end": end,
        "capacity": capacity,
        "used_bytes": used,
    }
    if 0 < used <= maximum_bytes:
        path = name + ".bin"
        if dump_raw(path, begin, used):
            record["path"] = path
    return record


def discover_vectors(label, address, size, metadata, maximum_bytes=128_000_000):
    try:
        data = memory(address, size)
    except gdb.MemoryError:
        return
    for offset in range(0, size - 23, 8):
        begin, end, capacity = struct.unpack_from("<QQQ", data, offset)
        used = end - begin
        allocated = capacity - begin
        if not (
            0x10000 <= begin <= end <= capacity
            and 0 < used <= allocated <= maximum_bytes
        ):
            continue
        widths = [
            width
            for width in (4, 8, 12, 16, 24, 32, 40, 48, 64, 72, 96, 104)
            if used % width == 0
        ]
        if not widths:
            continue
        path = "%s_%03x.bin" % (label, offset)
        if dump_raw(path, begin, used):
            metadata.append(
                {
                    "owner": label,
                    "offset": offset,
                    "begin": begin,
                    "end": end,
                    "capacity": capacity,
                    "used_bytes": used,
                    "possible_element_widths": widths,
                    "path": path,
                }
            )


BASE = image_base("surveyorslam_processing_node")
OUTPUT.mkdir(parents=True, exist_ok=True)
state = {
    "in_match": False,
    "match_thread": None,
    "match_seen": 0,
    "pending": None,
    "iterations": [],
}


def write_metadata():
    payload = {
        "executable_base": BASE,
        "target_match_index": TARGET_MATCH_INDEX,
        "match_offset": MATCH_OFFSET,
        "correspondence_offset": CORRESPONDENCE_OFFSET,
        "correspondence_return_offset": CORRESPONDENCE_RETURN_OFFSET,
        "iterations": state["iterations"],
        "target_objects": state.get("target_objects", []),
        "correspondence_finders": state.get("correspondence_finders", []),
        "solver_targets": state.get("solver_targets", {}),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


class MatchReturn(gdb.FinishBreakpoint):
    def stop(self):
        write_metadata()
        gdb.write("captured %d local ICP correspondence iterations\n" % len(state["iterations"]))
        gdb.execute("quit")
        return False


class MatchBreakpoint(gdb.Breakpoint):
    def stop(self):
        if state["in_match"]:
            return False
        match_index = state["match_seen"]
        state["match_seen"] += 1
        if match_index != TARGET_MATCH_INDEX:
            return False
        state["in_match"] = True
        state["match_thread"] = gdb.selected_thread().num
        dump_raw("matcher.bin", int(gdb.parse_and_eval("$rsi")), 0x300)
        dump_raw("source_owner.bin", int(gdb.parse_and_eval("$rdx")), 0x180)
        MatchReturn(gdb.newest_frame(), internal=True)
        return False


class CorrespondenceEntry(gdb.Breakpoint):
    def stop(self):
        if not state["in_match"] or gdb.selected_thread().num != state["match_thread"]:
            return False
        index = len(state["iterations"])
        registers = {
            name: int(gdb.parse_and_eval("$" + name))
            for name in ("rdi", "rsi", "rdx", "rcx", "r8")
        }
        prefix = "iteration_%02d" % index
        record = {
            "index": index,
            "binary_iteration": registers["r8"],
            "registers": registers,
            "transformed_source": dump_vector(prefix + "_source", registers["rsi"]),
            "correspondences_before": dump_vector(prefix + "_correspondences_before", registers["rdx"]),
        }
        dump_raw(prefix + "_search.bin", registers["rdi"], 0x400)
        dump_raw(prefix + "_context.bin", registers["rcx"], 0x180)
        if index == 0:
            matcher_fields = struct.unpack("<QQQ", memory(registers["rdi"], 24))
            targets = []
            for object_index, address in enumerate(matcher_fields):
                label = "matcher_target_%d" % object_index
                item = {"index": object_index, "address": address, "vectors": []}
                if address >= 0x10000 and dump_raw(label + ".bin", address, 0x2000):
                    discover_vectors(label, address, 0x2000, item["vectors"])
                targets.append(item)
            state["target_objects"] = targets
            correspondence_owner = matcher_fields[0]
            finder_begin, finder_end, _ = read_vector(correspondence_owner + 8)
            finders = []
            for finder_index, offset in enumerate(range(finder_begin, finder_end, 16)):
                address, control_block = struct.unpack("<QQ", memory(offset, 16))
                label = "correspondence_finder_%d" % finder_index
                item = {
                    "index": finder_index,
                    "address": address,
                    "control_block": control_block,
                    "vectors": [],
                }
                if address >= 0x10000 and dump_raw(label + ".bin", address, 0x4000):
                    discover_vectors(label, address, 0x4000, item["vectors"])
                    point_vector_owner = struct.unpack("<Q", memory(address + 0x38, 8))[0]
                    item["point_vector"] = dump_vector(
                        label + "_points", point_vector_owner
                    )
                finders.append(item)
            state["correspondence_finders"] = finders
            solver = matcher_fields[1]
            target_point_owner, target_normal_owner = struct.unpack(
                "<QQ", memory(solver + 8, 16)
            )
            state["solver_targets"] = {
                "point_owner": target_point_owner,
                "normal_owner": target_normal_owner,
                "points": dump_vector("solver_target_points", target_point_owner),
                "normals": dump_vector("solver_target_normals", target_normal_owner),
            }
        state["iterations"].append(record)
        state["pending"] = index
        return False


class CorrespondenceReturn(gdb.Breakpoint):
    def stop(self):
        if (
            not state["in_match"]
            or gdb.selected_thread().num != state["match_thread"]
            or state["pending"] is None
        ):
            return False
        index = state["pending"]
        record = state["iterations"][index]
        output_address = record["registers"]["rdx"]
        record["correspondences_after"] = dump_vector(
            "iteration_%02d_correspondences_after" % index, output_address
        )
        dump_raw(
            "iteration_%02d_correspondences_owner_after.bin" % index,
            output_address,
            0x40,
        )
        state["pending"] = None
        write_metadata()
        return False


MatchBreakpoint("*%#x" % (BASE + MATCH_OFFSET), internal=True)
CorrespondenceEntry("*%#x" % (BASE + CORRESPONDENCE_OFFSET), internal=True)
CorrespondenceReturn("*%#x" % (BASE + CORRESPONDENCE_RETURN_OFFSET), internal=True)
gdb.write("installed local ICP iteration capture at executable base %#x\n" % BASE)
end
continue
