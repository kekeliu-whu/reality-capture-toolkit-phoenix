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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_gravity_update.json"
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
RAW_VPTR = IMAGE_BASE + 0x1830660
armed = False
owner = 0


class Entry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x6e0280), internal=True)

    def stop(self):
        global armed, owner
        candidate = int(gdb.parse_and_eval("$rdi"))
        inferior = gdb.selected_inferior()
        vptr = struct.unpack("<Q", bytes(inferior.read_memory(candidate, 8)))[0]
        if vptr != RAW_VPTR or armed:
            return False
        owner = candidate
        sample = int(gdb.parse_and_eval("$rsi"))
        inferior.write_memory(
            owner + 0xb0,
            struct.pack(
                "<4d",
                0.09177029818504472,
                0.05333660875487905,
                -0.5892537473827747,
                0.8009453943833934,
            ),
        )
        inferior.write_memory(owner + 0xd0, struct.pack("<3d", 1.0, 2.0, 3.0))
        inferior.write_memory(sample, struct.pack("<3d", 4.0, 5.0, 6.0))
        gdb.execute("set $xmm0.v2_double[0] = 0.1")
        armed = True
        return False


class Result(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x6e06a7), internal=True)

    def stop(self):
        if not armed or int(gdb.parse_and_eval("$rbx")) != owner:
            return False
        inferior = gdb.selected_inferior()
        result = struct.unpack("<3d", bytes(inferior.read_memory(owner + 0xd0, 24)))
        OUTPUT.write_text(json.dumps({"gravity": result}, indent=2) + "\n")
        gdb.write("captured controlled raw IMU gravity update\n")
        gdb.execute("quit")
        return False


Entry()
Result()
end
continue
