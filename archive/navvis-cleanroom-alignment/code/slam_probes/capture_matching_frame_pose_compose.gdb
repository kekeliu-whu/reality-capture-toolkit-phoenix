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
CALL_INDEX = int(os.environ["NAVVIS_PROBE_CALL_INDEX"])
MATCH_OFFSET = 0x6C4660
COMPOSE_ENTRY_OFFSET = 0x4A18D7
NORMALIZED_READY_OFFSET = 0x4A1974


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


def rigid3(address):
    translation = struct.unpack("<3d", memory(address, 24))
    quaternion = struct.unpack("<4d", memory(address + 32, 32))
    return {
        "address": address,
        "translation": list(translation),
        "quaternion_xyzw": list(quaternion),
    }


match_count = 0
record = {"call": CALL_INDEX}


def write_output():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


class ComposeEntry(gdb.Breakpoint):
    def stop(self):
        if match_count != CALL_INDEX:
            return False
        record["inverse_matching_frame"] = rigid3(
            int(gdb.parse_and_eval("$rsi"))
        )
        record["predicted_local"] = rigid3(int(gdb.parse_and_eval("$rdx")))
        record["compose_output_address"] = int(gdb.parse_and_eval("$rdi"))
        write_output()
        return False


class NormalizedReady(gdb.Breakpoint):
    def stop(self):
        if match_count != CALL_INDEX:
            return False
        frame = gdb.newest_frame()
        base_pointer = int(gdb.parse_and_eval("$rbp"))
        record["normalized_initial"] = rigid3(base_pointer - 0x220)
        write_output()
        gdb.write("captured matching-frame pose composition for call %d\n" % CALL_INDEX)
        gdb.execute("quit")
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        return False


base = image_base("surveyorslam_processing_node")
ComposeEntry("*%#x" % (base + COMPOSE_ENTRY_OFFSET), internal=True)
NormalizedReady("*%#x" % (base + NORMALIZED_READY_OFFSET), internal=True)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
