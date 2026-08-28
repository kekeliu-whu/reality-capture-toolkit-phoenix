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
PAIR_SCORE_READY_OFFSET = 0x498AE7
AFTER_CALL = int(os.environ.get("NAVVIS_PROBE_AFTER_CALL", "818"))
REGION_RADIUS = float(os.environ.get("NAVVIS_PROBE_REGION_RADIUS", "0.12"))
CENTERS = tuple(
    tuple(float(component) for component in center.split(","))
    for center in os.environ["NAVVIS_PROBE_CENTERS"].split(";")
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
    return any(
        abs(center[0] - wanted[0]) <= REGION_RADIUS
        and abs(center[1] - wanted[1]) <= REGION_RADIUS
        and abs(center[2] - wanted[2]) <= REGION_RADIUS
        for center in (
            state["primary"]["center"], state["secondary"]["center"]
        )
        for wanted in CENTERS
    )


BASE = image_base("surveyorslam_processing_node")
match_count = 0
records = []


def write_output():
    OUTPUT.write_text(json.dumps(
        {
            "after_call": AFTER_CALL,
            "match_count": match_count,
            "centers": CENTERS,
            "region_radius": REGION_RADIUS,
            "records": records,
        },
        indent=2,
        sort_keys=True,
    ) + "\n")


class PairScoreReady(gdb.Breakpoint):
    def stop(self):
        stack_pointer = int(gdb.parse_and_eval("$rsp"))
        source_pointer = struct.unpack(
            "<Q", memory(stack_pointer + 8, 8)
        )[0]
        neighbor_pointer = int(gdb.parse_and_eval("$rbx"))
        source = voxel_state(source_pointer)
        neighbor = voxel_state(neighbor_pointer)
        if not (near_candidate(source) or near_candidate(neighbor)):
            return False
        score = float(gdb.parse_and_eval("$xmm10.v4_float[0]"))
        best_score = struct.unpack("<f", memory(stack_pointer + 0x14, 4))[0]
        source_center = (
            source["secondary"]["center"]
            if source["is_split"]
            and source["secondary"]["weight"] > source["primary"]["weight"]
            else source["primary"]["center"]
        )
        neighbor_center = (
            neighbor["secondary"]["center"]
            if neighbor["is_split"]
            and neighbor["secondary"]["weight"]
            > neighbor["primary"]["weight"]
            else neighbor["primary"]["center"]
        )
        delta = [
            neighbor_center[axis] - source_center[axis]
            for axis in range(3)
        ]
        distance_squared = (
            delta[2] * delta[2] + delta[1] * delta[1]
        ) + delta[0] * delta[0]
        records.append({
            "thread": gdb.selected_thread().num,
            "score": score,
            "best_score": best_score,
            "distance_squared_recomputed": distance_squared,
            "source": source,
            "neighbor": neighbor,
        })
        write_output()
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        call_index = match_count
        match_count += 1
        if call_index == AFTER_CALL:
            score_breakpoint.enabled = True
        elif call_index == AFTER_CALL + 1:
            score_breakpoint.enabled = False
            write_output()
            gdb.write(
                "captured %d regional pair scores after call %d\n"
                % (len(records), AFTER_CALL)
            )
            gdb.execute("quit")
        return False


score_breakpoint = PairScoreReady(
    "*%#x" % (BASE + PAIR_SCORE_READY_OFFSET), internal=True
)
score_breakpoint.enabled = False
LocalMatch("*%#x" % (BASE + MATCH_OFFSET), internal=True)
end
continue
