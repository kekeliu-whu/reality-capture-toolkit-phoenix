set pagination off
set confirm off
starti
python
import gdb
import json
from pathlib import Path
import struct


OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_merges.json")
MERGE_OFFSET = 0x49B530
CANDIDATE_OFFSET = 0x4987D0
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


def base_surfel(voxel, offset):
    raw = memory(voxel + offset, 104)
    return {
        "weight": struct.unpack_from("<f", raw, 8)[0],
        "count": struct.unpack_from("<I", raw, 12)[0],
        "center": list(struct.unpack_from("<3f", raw, 16)),
        "viewpoint_mean": list(struct.unpack_from("<3f", raw, 28)),
        "covariance_memory": list(struct.unpack_from("<9f", raw, 40)),
        "normal": list(struct.unpack_from("<3f", raw, 76)),
        "dirty": bool(raw[100]),
    }


def voxel_state(voxel):
    return {
        "address": voxel,
        "primary": base_surfel(voxel, 8),
        "secondary": base_surfel(voxel, 0x70),
        "secondary_active": bool(memory(voxel + 0xE4, 1)[0]),
    }


def key_vector(vector):
    begin, end = struct.unpack("<QQ", memory(vector, 16))
    count = (end - begin) // 12
    raw = memory(begin, count * 12)
    return [list(struct.unpack_from("<3i", raw, 12 * index)) for index in range(count)]


records = []
candidate_records = []


class SurfelCandidate(gdb.Breakpoint):
    def stop(self):
        grid = int(gdb.parse_and_eval("$rdi"))
        voxel = int(gdb.parse_and_eval("$rsi"))
        state = voxel_state(voxel)
        center = state["primary"]["center"]
        count = state["primary"]["count"]
        if (
            count <= 30
            and -0.6 <= center[0] <= 0.9
            and -2.8 <= center[1] <= -0.8
            and 0.65 <= center[2] <= 1.15
        ):
            candidate_records.append(
                {
                    "thread": gdb.selected_thread().num,
                    "grid": grid,
                    "voxel": state,
                    "neighbor_keys": key_vector(int(gdb.parse_and_eval("$rdx"))),
                }
            )
        return False


class SurfelMerge(gdb.Breakpoint):
    def stop(self):
        grid = int(gdb.parse_and_eval("$rdi"))
        first = int(gdb.parse_and_eval("$rsi"))
        second = int(gdb.parse_and_eval("$rdx"))
        records.append(
            {
                "thread": gdb.selected_thread().num,
                "grid": grid,
                "grid_head_hex": memory(grid, 0xD8).hex(),
                "first": voxel_state(first),
                "second": voxel_state(second),
            }
        )
        return False


class FirstMatch(gdb.Breakpoint):
    def stop(self):
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(
                {"candidates": candidate_records, "merges": records},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        gdb.write(
            "captured %d regional candidates and %d surfel merges\n"
            % (len(candidate_records), len(records))
        )
        gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
SurfelCandidate("*%#x" % (base + CANDIDATE_OFFSET), internal=True)
SurfelMerge("*%#x" % (base + MERGE_OFFSET), internal=True)
FirstMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
