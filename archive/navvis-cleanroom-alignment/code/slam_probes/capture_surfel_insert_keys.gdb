set pagination off
set confirm off
starti
python
import gdb
import json
from pathlib import Path
import math
import struct


OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_insert_keys.json")
INSERT_OFFSET = 0x499970
MATCH_OFFSET = 0x6C4660
WATCHED_KEYS = {
    (-4, -20, 9),
    (-4, -10, 10),
    (-4, -10, 11),
    (-3, -21, 8),
    (0, -27, 8),
    (0, -22, 8),
    (2, -23, 8),
    (2, -17, 9),
    (2, -17, 10),
    (4, -23, 9),
    (4, -18, 9),
    (4, -18, 10),
    (6, -24, 8),
    (8, -25, 9),
}


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


records = []
fine_insertions = 0


class SurfelInsert(gdb.Breakpoint):
    def stop(self):
        global fine_insertions
        grid = int(gdb.parse_and_eval("$rdi"))
        origin_x, origin_y, origin_z, inverse = struct.unpack(
            "<4d", memory(grid + 0x20, 0x20)
        )
        if inverse != 10.0:
            return False
        fine_insertions += 1
        ray = int(gdb.parse_and_eval("$rsi"))
        values = struct.unpack("<6f", memory(ray, 24))
        key = (
            math.floor((values[3] + origin_x) * inverse),
            math.floor((values[4] + origin_y) * inverse),
            math.floor((values[5] + origin_z) * inverse),
        )
        if key in WATCHED_KEYS:
            records.append(
                {
                    "key": list(key),
                    "origin": list(values[:3]),
                    "point": list(values[3:]),
                    "grid": grid,
                    "thread": gdb.selected_thread().num,
                }
            )
        return False


class FirstMatch(gdb.Breakpoint):
    def stop(self):
        payload = {"fine_insertions": fine_insertions, "records": records}
        OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        gdb.write(
            "captured %d watched fine-grid insertions of %d total\n"
            % (len(records), fine_insertions)
        )
        gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
SurfelInsert("*%#x" % (base + INSERT_OFFSET), internal=True)
FirstMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
