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
    "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_merge_candidates_first4.json",
))
CANDIDATES_READY_OFFSETS = (0x49C2E1, 0x49C99D)
MATCH_OFFSET = 0x6C4660
MATCH_LIMIT = int(os.environ.get("NAVVIS_PROBE_MATCH_LIMIT", "4"))


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


base = image_base("surveyorslam_processing_node")
records = []
match_count = 0


def write_output():
    OUTPUT.write_text(json.dumps({
        "match_count": match_count,
        "records": records,
    }, indent=2, sort_keys=True) + "\n")


class CandidatesReady(gdb.Breakpoint):
    def __init__(self, offset):
        super().__init__("*%#x" % (base + offset), internal=True)
        self.offset = offset

    def stop(self):
        grid = int(gdb.parse_and_eval("$rbp"))
        stack = int(gdb.parse_and_eval("$rsp"))
        begin, end = struct.unpack("<QQ", memory(stack + 0x10, 16))
        count = (end - begin) // 24
        candidates = []
        for position in range(count):
            first, second = struct.unpack(
                "<QQ", memory(begin + 24 * position, 16)
            )
            valid = memory(begin + 24 * position + 16, 1)[0] != 0
            if valid:
                candidates.append({
                    "position": position,
                    "first": first,
                    "second": second,
                })
        cell_begin, cell_end = struct.unpack("<QQ", memory(grid + 0x80, 16))
        record = {
            "before_match": match_count,
            "ready_offset": self.offset,
            "grid": grid,
            "cell_count": (cell_end - cell_begin) // 232,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        if self.offset == 0x49C99D:
            active_begin, active_end = struct.unpack(
                "<QQ", memory(grid + 0x98, 16)
            )
            record["active_indices"] = list(struct.unpack(
                "<%dQ" % ((active_end - active_begin) // 8),
                memory(active_begin, active_end - active_begin),
            ))
        records.append(record)
        write_output()
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        write_output()
        if match_count >= MATCH_LIMIT:
            gdb.write(
                "captured %d candidate arrays before %d matches\n"
                % (len(records), MATCH_LIMIT)
            )
            gdb.execute("quit")
        return False


for candidate_offset in CANDIDATES_READY_OFFSETS:
    CandidatesReady(candidate_offset)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
