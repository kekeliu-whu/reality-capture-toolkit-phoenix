set pagination off
set confirm off
starti
python
import gdb
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ["NAVVIS_PROBE_OUTPUT"])
MATCH_OFFSET = 0x6C4660
MERGE_DRIVER_OFFSET = 0x4987D0
MERGE_OFFSET = 0x49B530
AFTER_CALL = int(os.environ.get("NAVVIS_PROBE_AFTER_CALL", "259"))
REGION_RADIUS = float(os.environ.get("NAVVIS_PROBE_REGION_RADIUS", "0.15"))
CENTERS = tuple(
    tuple(float(component) for component in center.split(","))
    for center in os.environ["NAVVIS_PROBE_CENTERS"].split(";")
)


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


def base_surfel(raw, offset):
    return {
        "weight": struct.unpack_from("<f", raw, offset + 8)[0],
        "count": struct.unpack_from("<I", raw, offset + 12)[0],
        "center": list(struct.unpack_from("<3f", raw, offset + 16)),
        "covariance_memory": list(struct.unpack_from("<9f", raw, offset + 28)),
        "viewpoint_mean": list(struct.unpack_from("<3f", raw, offset + 64)),
        "normal": list(struct.unpack_from("<3f", raw, offset + 76)),
        "eigenvalues": list(struct.unpack_from("<3f", raw, offset + 88)),
        "dirty": bool(raw[offset + 100]),
    }


def voxel_state(voxel):
    raw = memory(voxel, 232)
    return {
        "address": voxel,
        "primary": base_surfel(raw, 8),
        "secondary": base_surfel(raw, 112),
        "split_normal": list(struct.unpack_from("<3f", raw, 216)),
        "is_split": bool(raw[228]),
    }


def near_candidate(state):
    centers = (state["primary"]["center"], state["secondary"]["center"])
    return any(
        abs(center[0] - wanted[0]) <= REGION_RADIUS
        and abs(center[1] - wanted[1]) <= REGION_RADIUS
        and abs(center[2] - wanted[2]) <= REGION_RADIUS
        for center in centers
        for wanted in CENTERS
    )


BASE = image_base("surveyorslam_processing_node")
match_count = 0
drivers = []
merges = []


def write_output():
    OUTPUT.write_text(json.dumps({
        "after_call": AFTER_CALL,
        "match_count": match_count,
        "centers": CENTERS,
        "region_radius": REGION_RADIUS,
        "drivers": drivers,
        "merges": merges,
    }, indent=2, sort_keys=True) + "\n")


class MergeDriver(gdb.Breakpoint):
    def stop(self):
        voxel = int(gdb.parse_and_eval("$rsi"))
        state = voxel_state(voxel)
        if near_candidate(state):
            begin, end, _ = struct.unpack(
                "<QQQ", memory(int(gdb.parse_and_eval("$rdx")), 24)
            )
            neighbor_count = (end - begin) // 12
            drivers.append({
                "thread": gdb.selected_thread().num,
                "grid": int(gdb.parse_and_eval("$rdi")),
                "voxel": state,
                "neighbor_keys": [
                    list(struct.unpack("<3i", memory(begin + 12 * index, 12)))
                    for index in range(neighbor_count)
                ],
            })
            write_output()
        return False


class SurfelMerge(gdb.Breakpoint):
    def stop(self):
        first_pointer = int(gdb.parse_and_eval("$rsi"))
        second_pointer = int(gdb.parse_and_eval("$rdx"))
        caller_frame = gdb.newest_frame().older()
        caller_pc = int(caller_frame.pc()) if caller_frame is not None else 0
        first = voxel_state(first_pointer)
        second = voxel_state(second_pointer)
        if near_candidate(first) or near_candidate(second):
            record = {
                "thread": gdb.selected_thread().num,
                "grid": int(gdb.parse_and_eval("$rdi")),
                "caller_pc": caller_pc,
                "caller_offset": caller_pc - BASE,
                "first": first,
                "second": second,
            }
            merges.append(record)
            write_output()
            MergeReturn(record, first_pointer, second_pointer)
        return False


class MergeReturn(gdb.FinishBreakpoint):
    def __init__(self, record, first_pointer, second_pointer):
        super().__init__(gdb.newest_frame(), internal=True)
        self.record = record
        self.first_pointer = first_pointer
        self.second_pointer = second_pointer

    def stop(self):
        self.record["first_after"] = voxel_state(self.first_pointer)
        self.record["second_after"] = voxel_state(self.second_pointer)
        write_output()
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        call_index = match_count
        match_count += 1
        if call_index == AFTER_CALL:
            driver_breakpoint.enabled = True
            merge_breakpoint.enabled = True
        elif call_index == AFTER_CALL + 1:
            driver_breakpoint.enabled = False
            merge_breakpoint.enabled = False
            write_output()
            gdb.write(
                "captured %d candidate drivers and %d merges after call %d\n"
                % (len(drivers), len(merges), AFTER_CALL)
            )
            gdb.execute("quit")
        return False


driver_breakpoint = MergeDriver("*%#x" % (BASE + MERGE_DRIVER_OFFSET), internal=True)
merge_breakpoint = SurfelMerge("*%#x" % (BASE + MERGE_OFFSET), internal=True)
driver_breakpoint.enabled = False
merge_breakpoint.enabled = False
LocalMatch("*%#x" % (BASE + MATCH_OFFSET), internal=True)
end
continue
