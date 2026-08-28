set pagination off
set confirm off
starti
python
import gdb
import os
from pathlib import Path
import struct

OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_tracker_raw.bin"
))
MATCH_LIMIT = int(os.environ.get("NAVVIS_PROBE_MATCH_LIMIT", "1"))


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


IMAGE_BASE = image_base("surveyorslam_processing_node")
RAW_VPTR = IMAGE_BASE + 0x1830660
ADVANCE_OFFSET = 0x6e0f30
ADD_DATA_OFFSET = 0x6e2670
MATCH_OFFSET = 0x6c4660

with OUTPUT.open("wb") as stream:
    stream.write(b"NVIMUTR1")

def append_record(kind, payload):
    with OUTPUT.open("ab") as stream:
        stream.write(struct.pack("<BI", kind, len(payload)))
        stream.write(payload)

class AdvanceFinish(gdb.FinishBreakpoint):
    def __init__(self, owner, target_time):
        super().__init__(internal=True)
        self.owner = owner
        self.target_time = target_time

    def stop(self):
        inferior = gdb.selected_inferior()
        state = bytes(inferior.read_memory(self.owner + 0x48, 0xa0))
        append_record(3, struct.pack("<QQ", self.owner, self.target_time) + state)
        return False

class AdvanceBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + ADVANCE_OFFSET), internal=True)
        self.hits = 0

    def stop(self):
        owner = int(gdb.parse_and_eval("$rdi"))
        target_time = int(gdb.parse_and_eval("$rsi")) & 0xffffffffffffffff
        inferior = gdb.selected_inferior()
        vptr = struct.unpack("<Q", bytes(inferior.read_memory(owner, 8)))[0]
        if vptr != RAW_VPTR:
            return False
        state = bytes(inferior.read_memory(owner + 0x48, 0xa0))
        append_record(2, struct.pack("<QQ", owner, target_time) + state)
        AdvanceFinish(owner, target_time)
        self.hits += 1
        if self.hits % 100 == 0:
            gdb.write("IMU_RAW advance hits=%d target=%d\n" % (self.hits, target_time))
        return False

class AddDataBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + ADD_DATA_OFFSET), internal=True)
        self.hits = 0

    def stop(self):
        owner = int(gdb.parse_and_eval("$rdi"))
        sample = int(gdb.parse_and_eval("$rsi"))
        inferior = gdb.selected_inferior()
        vptr = struct.unpack("<Q", bytes(inferior.read_memory(owner, 8)))[0]
        if vptr != RAW_VPTR:
            return False
        payload = bytes(inferior.read_memory(sample, 0x90))
        append_record(1, struct.pack("<QQ", owner, sample) + payload)
        self.hits += 1
        if self.hits % 100 == 0:
            gdb.write("IMU_RAW add hits=%d\n" % self.hits)
        return False

AdvanceBreakpoint()
AddDataBreakpoint()


class LocalMatch(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + MATCH_OFFSET), internal=True)
        self.hits = 0

    def stop(self):
        self.hits += 1
        if self.hits >= MATCH_LIMIT:
            gdb.write("captured raw IMU tracker through %d local matches\n" % self.hits)
            gdb.execute("quit")
        return False


LocalMatch()
end
continue
