set pagination off
set confirm off
starti
python
import gdb
import hashlib
import json
from pathlib import Path
import struct


OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_batches_until_match")
WORKER_OFFSET = 0x48ED50
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


OUTPUT.mkdir(parents=True, exist_ok=True)
records = []
saved = {}


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
            path = "batch_%02d.bin" % len(saved)
            (OUTPUT / path).write_bytes(rays)
            saved[digest] = path
        records.append(
            {
                "thread": gdb.selected_thread().num,
                "worker": worker,
                "caller": caller,
                "caller_offset": caller - base,
                "worker_head_hex": memory(worker, 128).hex(),
                "vector_owner": vector_owner,
                "count": count,
                "sha256": digest,
                "path": saved[digest],
            }
        )
        return False


class FirstMatch(gdb.Breakpoint):
    def stop(self):
        (OUTPUT / "metadata.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n"
        )
        gdb.write(
            "captured %d worker calls and %d unique batches\n"
            % (len(records), len(saved))
        )
        gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
SurfelWorker("*%#x" % (base + WORKER_OFFSET), internal=True)
FirstMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
