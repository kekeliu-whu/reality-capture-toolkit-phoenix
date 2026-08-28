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
    "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_cells_at_candidates.json",
))
CANDIDATES_READY_OFFSETS = (0x49C2E1, 0x49C99D)
MATCH_OFFSET = 0x6C4660
MATCH_LIMIT = int(os.environ.get("NAVVIS_PROBE_MATCH_LIMIT", "5"))
CELL_INDICES = tuple(int(value) for value in os.environ.get(
    "NAVVIS_PROBE_CELL_INDICES", "9612,9615"
).split(",") if value)


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


def voxel_fields(cell_begin, index):
    raw = memory(cell_begin + 232 * index, 232)
    return {
        "index": index,
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
        "cell_indices": CELL_INDICES,
        "records": records,
    }, indent=2, sort_keys=True) + "\n")


class CandidatesReady(gdb.Breakpoint):
    def __init__(self, offset):
        super().__init__("*%#x" % (base + offset), internal=True)
        self.offset = offset

    def stop(self):
        grid = int(gdb.parse_and_eval("$rbp"))
        cell_begin, cell_end = struct.unpack("<QQ", memory(grid + 0x80, 16))
        cell_count = (cell_end - cell_begin) // 232
        record = {
            "before_match": match_count,
            "ready_offset": self.offset,
            "cell_count": cell_count,
            "cells": [
                voxel_fields(cell_begin, index)
                for index in CELL_INDICES if index < cell_count
            ],
        }
        if self.offset == 0x49C99D:
            active_begin, active_end = struct.unpack("<QQ", memory(grid + 0x98, 16))
            active = list(struct.unpack(
                "<%dQ" % ((active_end - active_begin) // 8),
                memory(active_begin, active_end - active_begin),
            ))
            record["active_positions"] = {
                str(index): active.index(index)
                for index in CELL_INDICES if index in active
            }
        records.append(record)
        write_output()
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        write_output()
        if match_count >= MATCH_LIMIT:
            gdb.write("captured candidate cell states before %d matches\n" % MATCH_LIMIT)
            gdb.execute("quit")
        return False


for candidate_offset in CANDIDATES_READY_OFFSETS:
    CandidatesReady(candidate_offset)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
