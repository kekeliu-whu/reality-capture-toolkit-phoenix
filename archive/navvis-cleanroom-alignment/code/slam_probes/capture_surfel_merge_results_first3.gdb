set pagination off
set confirm off
starti
python
import gdb
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT",
    "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_merge_results_first3.json",
))
MERGE_OFFSET = 0x49B530
MATCH_OFFSET = 0x6C4660
MATCH_LIMIT = int(os.environ.get("NAVVIS_PROBE_MATCH_LIMIT", "3"))
MATCH_START = int(os.environ.get("NAVVIS_PROBE_MATCH_START", "0"))
CENTER = tuple(float(value) for value in os.environ.get(
    "NAVVIS_PROBE_CENTER", "nan,nan,nan"
).split(","))
CENTER_RADIUS = float(os.environ.get("NAVVIS_PROBE_CENTER_RADIUS", "inf"))


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


def voxel_fields(voxel):
    raw = memory(voxel, 232)
    return {
        "address": voxel,
        "primary": base_surfel(raw, 8),
        "secondary": base_surfel(raw, 112),
        "split_normal": list(struct.unpack_from("<3f", raw, 216)),
        "is_split": bool(raw[228]),
    }


base = image_base("surveyorslam_processing_node")
records = []
match_count = 0


def write_output():
    OUTPUT.write_text(json.dumps({
        "match_count": match_count,
        "match_start": MATCH_START,
        "center": CENTER,
        "center_radius": CENTER_RADIUS,
        "merges": records,
    }, indent=2, sort_keys=True) + "\n")


def near_probe_center(voxel):
    if not all(value == value for value in CENTER):
        return True
    for name in ("primary", "secondary"):
        surfel = voxel[name]
        if surfel["weight"] == 0.0:
            continue
        if all(abs(value - expected) <= CENTER_RADIUS
               for value, expected in zip(surfel["center"], CENTER)):
            return True
    return False


class MergeReturn(gdb.FinishBreakpoint):
    def __init__(self, record, first, second):
        super().__init__(gdb.newest_frame(), internal=True)
        self.record = record
        self.first = first
        self.second = second

    def stop(self):
        self.record["after"] = {
            "first": voxel_fields(self.first),
            "second": voxel_fields(self.second),
        }
        write_output()
        return False


class SurfelMerge(gdb.Breakpoint):
    def stop(self):
        first = int(gdb.parse_and_eval("$rsi"))
        second = int(gdb.parse_and_eval("$rdx"))
        first_before = voxel_fields(first)
        second_before = voxel_fields(second)
        if not (near_probe_center(first_before) or
                near_probe_center(second_before)):
            return False
        record = {
            "before_match": match_count,
            "grid": int(gdb.parse_and_eval("$rdi")),
            "cell_data": struct.unpack(
                "<Q", memory(int(gdb.parse_and_eval("$rdi")) + 0x80, 8)
            )[0],
            "before": {
                "first": first_before,
                "second": second_before,
            },
        }
        records.append(record)
        write_output()
        MergeReturn(record, first, second)
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        if match_count >= MATCH_START:
            merge_breakpoint.enabled = True
        write_output()
        if match_count >= MATCH_LIMIT:
            gdb.write(
                "captured %d surfel merge results before %d matches\n"
                % (len(records), MATCH_LIMIT)
            )
            gdb.execute("quit")
        return False


merge_breakpoint = SurfelMerge("*%#x" % (base + MERGE_OFFSET), internal=True)
merge_breakpoint.enabled = MATCH_START == 0
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
