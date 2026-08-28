set pagination off
set confirm off
starti
python
import gdb
import os
import struct

OUTPUT = os.environ.get(
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_accumulated_ranges.bin"
)
TARGET_INDEX = int(os.environ.get("NAVVIS_PROBE_TARGET_INDEX", "-1"))
TARGET_END_INDEX = int(os.environ.get(
    "NAVVIS_PROBE_TARGET_END_INDEX", str(TARGET_INDEX)
))


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


IMAGE_BASE = image_base("surveyorslam_processing_node")

with open(OUTPUT, "wb") as stream:
    stream.write(b"NVACCUM1")

class AccumulatedRangeBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x4af860), internal=True)
        self.hits = 0

    def stop(self):
        hit = self.hits
        self.hits += 1
        if TARGET_INDEX >= 0 and not (TARGET_INDEX <= hit <= TARGET_END_INDEX):
            return False
        owner = int(gdb.parse_and_eval("$rdi"))
        vector = int(gdb.parse_and_eval("$rsi"))
        inferior = gdb.selected_inferior()
        begin, end, capacity = struct.unpack(
            "<QQQ", bytes(inferior.read_memory(vector, 24))
        )
        count = (end - begin) // 32
        records = bytes(inferior.read_memory(begin, count * 32))
        with open(OUTPUT, "ab") as stream:
            stream.write(struct.pack("<QQII", owner, vector, count, len(records)))
            stream.write(records)
        gdb.write(
            "ACCUM hit=%d owner=%#x records=%d begin=%#x end=%#x\n"
            % (hit, owner, count, begin, end)
        )
        if (
            (TARGET_INDEX >= 0 and hit >= TARGET_END_INDEX)
            or (TARGET_INDEX < 0 and self.hits >= 3)
        ):
            gdb.execute("quit")
        return False

AccumulatedRangeBreakpoint()
end
continue
