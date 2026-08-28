set pagination off
set confirm off
starti
python
import gdb
import hashlib
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT",
    "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_batches_first3",
))
WORKER_OFFSET = 0x48ED50
MATCH_OFFSET = 0x6C4660
MATCH_LIMIT = int(os.environ.get("NAVVIS_PROBE_MATCH_LIMIT", "3"))


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


OUTPUT.mkdir(parents=True, exist_ok=True)
records = []
saved = {}
match_count = 0


def write_metadata():
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "match_count": match_count,
        "records": records,
        "unique_batches": list(saved.values()),
    }, indent=2, sort_keys=True) + "\n")


class SurfelWorker(gdb.Breakpoint):
    def stop(self):
        worker = int(gdb.parse_and_eval("$rdi"))
        stack = int(gdb.parse_and_eval("$rsp"))
        caller = struct.unpack("<Q", memory(stack, 8))[0]
        vector_owner = struct.unpack("<Q", memory(worker, 8))[0]
        begin, end = struct.unpack("<QQ", memory(vector_owner, 16))
        count = (end - begin) // 24
        rays = memory(begin, count * 24)
        digest = hashlib.sha256(rays).hexdigest()
        if digest not in saved:
            batch_index = len(saved)
            path = "batch_%02d.bin" % batch_index
            (OUTPUT / path).write_bytes(rays)
            saved[digest] = {
                "index": batch_index,
                "before_match": match_count,
                "count": count,
                "sha256": digest,
                "path": path,
            }
        records.append({
            "before_match": match_count,
            "thread": gdb.selected_thread().num,
            "caller": caller,
            "caller_offset": caller - base,
            "count": count,
            "sha256": digest,
        })
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        write_metadata()
        if match_count >= MATCH_LIMIT:
            gdb.write(
                "captured %d worker calls and %d unique batches before %d matches\n"
                % (len(records), len(saved), match_count)
            )
            gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
SurfelWorker("*%#x" % (base + WORKER_OFFSET), internal=True)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
