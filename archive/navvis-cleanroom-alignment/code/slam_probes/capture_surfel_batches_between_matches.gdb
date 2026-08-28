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
WORKER_OFFSET = 0x48ED50
MATCH_OFFSET = 0x6C4660
START_BEFORE_MATCH = int(os.environ.get("NAVVIS_PROBE_MATCH_START", "933"))
END_BEFORE_MATCH = int(os.environ.get("NAVVIS_PROBE_MATCH_END", "933"))


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


OUTPUT.mkdir(parents=True, exist_ok=True)
records = []
match_count = 0


def write_metadata():
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "start_before_match": START_BEFORE_MATCH,
        "end_before_match": END_BEFORE_MATCH,
        "match_count": match_count,
        "records": records,
    }, indent=2, sort_keys=True) + "\n")


class SurfelWorker(gdb.Breakpoint):
    def stop(self):
        if not START_BEFORE_MATCH <= match_count <= END_BEFORE_MATCH:
            return False
        worker = int(gdb.parse_and_eval("$rdi"))
        stack = int(gdb.parse_and_eval("$rsp"))
        caller = struct.unpack("<Q", memory(stack, 8))[0]
        vector_owner = struct.unpack("<Q", memory(worker, 8))[0]
        begin, end = struct.unpack("<QQ", memory(vector_owner, 16))
        count = (end - begin) // 24
        rays = memory(begin, count * 24)
        digest = hashlib.sha256(rays).hexdigest()
        path = "match_%04d_worker_%02d.bin" % (match_count, len(records))
        (OUTPUT / path).write_bytes(rays)
        records.append({
            "before_match": match_count,
            "thread": gdb.selected_thread().num,
            "worker": worker,
            "caller": caller,
            "caller_offset": caller - base,
            "worker_head_hex": memory(worker, 128).hex(),
            "vector_owner": vector_owner,
            "count": count,
            "sha256": digest,
            "path": path,
        })
        write_metadata()
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        call_index = match_count
        match_count += 1
        if call_index > END_BEFORE_MATCH:
            write_metadata()
            gdb.write(
                "captured %d surfel batches around matches %d..%d\n"
                % (len(records), START_BEFORE_MATCH, END_BEFORE_MATCH)
            )
            gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
SurfelWorker("*%#x" % (base + WORKER_OFFSET), internal=True)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
