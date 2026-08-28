set pagination off
set confirm off
starti
python
import gdb
import json
from pathlib import Path
import struct


OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_merges_full_first3.json")
MERGE_OFFSET = 0x49B530
MATCH_OFFSET = 0x6C4660


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


class SurfelMerge(gdb.Breakpoint):
    def stop(self):
        records.append({
            "before_match": match_count,
            "grid": int(gdb.parse_and_eval("$rdi")),
            "first": voxel_fields(int(gdb.parse_and_eval("$rsi"))),
            "second": voxel_fields(int(gdb.parse_and_eval("$rdx"))),
        })
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        if match_count >= 3:
            OUTPUT.write_text(json.dumps({
                "match_count": match_count,
                "merges": records,
            }, indent=2, sort_keys=True) + "\n")
            gdb.write("captured %d surfel merge calls before three matches\n" % len(records))
            gdb.execute("quit")
        return False


SurfelMerge("*%#x" % (base + MERGE_OFFSET), internal=True)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
