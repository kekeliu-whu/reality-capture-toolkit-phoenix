set pagination off
set confirm off
set print thread-events off
starti
python
import gdb
import hashlib
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ["NAVVIS_PROBE_OUTPUT"])
MATCH_OFFSET = 0x6C4660
INSERT_LAUNCHER_OFFSET = 0x48F200
CAPTURE_BEFORE_MATCH = int(os.environ["NAVVIS_PROBE_MATCH_INDEX"])


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


def pointer(address):
    return struct.unpack("<Q", memory(address, 8))[0]


def deque_entries(owner, offset):
    # libstdc++ deque iterators contain cur/first/last/node. The grid stores
    # 16-byte polymorphic level entries, so one 512-byte block holds 32.
    cur, first, last, node = struct.unpack("<QQQQ", memory(owner + offset, 32))
    end_cur, _, _, end_node = struct.unpack(
        "<QQQQ", memory(owner + offset + 32, 32)
    )
    entries = []
    while node != end_node or cur != end_cur:
        if cur == last:
            node += 8
            first = pointer(node)
            last = first + 512
            cur = first
        entries.append(memory(cur, 16))
        cur += 16
    return entries


OUTPUT.mkdir(parents=True, exist_ok=True)
match_count = 0
maps = []


class InsertionLauncher(gdb.Breakpoint):
    def stop(self):
        if match_count != CAPTURE_BEFORE_MATCH:
            return False
        insertion_map = int(gdb.parse_and_eval("$rdi"))
        if insertion_map not in maps:
            maps.append(insertion_map)
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        call_index = match_count
        if call_index == CAPTURE_BEFORE_MATCH:
            records = []
            for map_ordinal, insertion_map in enumerate(maps):
                levels = []
                for level_index, entry in enumerate(
                    deque_entries(insertion_map, 0x78)
                ):
                    grid = struct.unpack_from("<Q", entry)[0]
                    begin, end = struct.unpack("<QQ", memory(grid + 0x80, 16))
                    count = (end - begin) // 232
                    cells = memory(begin, count * 232)
                    filename = (
                        f"map_{map_ordinal:02d}_level_{level_index:02d}_cells.bin"
                    )
                    (OUTPUT / filename).write_bytes(cells)
                    levels.append({
                        "index": level_index,
                        "grid": grid,
                        "grid_head_hex": memory(grid, 0xD8).hex(),
                        "cell_count": count,
                        "sha256": hashlib.sha256(cells).hexdigest(),
                        "path": filename,
                    })
                records.append({
                    "ordinal": map_ordinal,
                    "map": insertion_map,
                    "map_head_hex": memory(insertion_map, 0xD8).hex(),
                    "levels": levels,
                })
            (OUTPUT / "metadata.json").write_text(json.dumps({
                "before_match": CAPTURE_BEFORE_MATCH,
                "maps": records,
            }, indent=2, sort_keys=True) + "\n")
            gdb.write(
                "captured %d insertion maps before match %d\n"
                % (len(records), CAPTURE_BEFORE_MATCH)
            )
            gdb.execute("quit")
        match_count += 1
        return False


base = image_base("surveyorslam_processing_node")
InsertionLauncher("*%#x" % (base + INSERT_LAUNCHER_OFFSET), internal=True)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
