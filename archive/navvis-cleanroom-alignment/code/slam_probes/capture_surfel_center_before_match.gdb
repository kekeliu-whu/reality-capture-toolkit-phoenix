set pagination off
set confirm off
set print thread-events off
starti
python
import gdb
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ["NAVVIS_PROBE_OUTPUT"])
MATCH_OFFSET = 0x6C4660
CANDIDATES_READY_OFFSETS = (0x49C2E1, 0x49C99D)
CAPTURE_AFTER_CALL = int(os.environ.get("NAVVIS_PROBE_AFTER_CALL", "942"))
CAPTURE_START_CALL = int(os.environ.get(
    "NAVVIS_PROBE_CALL_START", str(CAPTURE_AFTER_CALL)
))
CAPTURE_END_CALL = int(os.environ.get(
    "NAVVIS_PROBE_CALL_END", str(CAPTURE_START_CALL + 1)
))
TARGET_CENTER = tuple(float(value) for value in os.environ.get(
    "NAVVIS_PROBE_CENTER", "3.0772114,2.8984168,0.2047006"
).split(","))
TOLERANCE = float(os.environ.get("NAVVIS_PROBE_CENTER_TOLERANCE", "1e-5"))
TARGET_INDEX_TEXT = os.environ.get("NAVVIS_PROBE_CELL_INDEX")
TARGET_INDEX = (
    int(TARGET_INDEX_TEXT) if TARGET_INDEX_TEXT is not None else None
)


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if (
                len(fields) >= 6
                and fields[2] == "00000000"
                and fragment in fields[-1]
            ):
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


def center_matches(center):
    return all(
        abs(center[axis] - TARGET_CENTER[axis]) <= TOLERANCE
        for axis in range(3)
    )


BASE = image_base("surveyorslam_processing_node")
match_count = 0
records = []


def write_output():
    OUTPUT.write_text(json.dumps(
        {
            "capture_after_call": CAPTURE_AFTER_CALL,
            "capture_start_call": CAPTURE_START_CALL,
            "capture_end_call": CAPTURE_END_CALL,
            "match_count": match_count,
            "target_center": TARGET_CENTER,
            "target_index": TARGET_INDEX,
            "tolerance": TOLERANCE,
            "records": records,
        },
        indent=2,
        sort_keys=True,
    ) + "\n")


class CandidatesReady(gdb.Breakpoint):
    def __init__(self, offset):
        super().__init__("*%#x" % (BASE + offset), internal=True)
        self.offset = offset

    def stop(self):
        grid = int(gdb.parse_and_eval("$rbp"))
        cell_begin, cell_end = struct.unpack("<QQ", memory(grid + 0x80, 16))
        cell_count = (cell_end - cell_begin) // 232
        raw_cells = memory(cell_begin, cell_count * 232)
        for index in range(cell_count):
            raw = raw_cells[index * 232:(index + 1) * 232]
            primary_center = struct.unpack_from("<3f", raw, 24)
            secondary_center = struct.unpack_from("<3f", raw, 128)
            if TARGET_INDEX is not None:
                matches = index == TARGET_INDEX
            else:
                matches = (
                    center_matches(primary_center)
                    or center_matches(secondary_center)
                )
            if not matches:
                continue
            records.append({
                "call_index": match_count - 1,
                "ready_offset": self.offset,
                "grid": grid,
                "cell_count": cell_count,
                "cell_index": index,
                "primary": base_surfel(raw, 8),
                "secondary": base_surfel(raw, 112),
                "split_normal": list(struct.unpack_from("<3f", raw, 216)),
                "is_split": bool(raw[228]),
            })
        write_output()
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        call_index = match_count
        match_count += 1
        if call_index == CAPTURE_START_CALL:
            for breakpoint in candidate_breakpoints:
                breakpoint.enabled = True
        elif call_index == CAPTURE_END_CALL:
            for breakpoint in candidate_breakpoints:
                breakpoint.enabled = False
            write_output()
            gdb.write(
                "captured %d matching surfel snapshots before call %d\n"
                % (len(records), call_index)
            )
            gdb.execute("quit")
        return False


candidate_breakpoints = [
    CandidatesReady(offset) for offset in CANDIDATES_READY_OFFSETS
]
for breakpoint in candidate_breakpoints:
    breakpoint.enabled = False
LocalMatch("*%#x" % (BASE + MATCH_OFFSET), internal=True)
end
continue
