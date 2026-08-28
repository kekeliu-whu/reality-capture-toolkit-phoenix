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


OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_correction_window.json"
))
START_NS = int(os.environ["NAVVIS_PROBE_START_NS"])
END_NS = int(os.environ["NAVVIS_PROBE_END_NS"])
ACTIVATE_NS = int(os.environ.get("NAVVIS_PROBE_ACTIVATE_NS", "0"))


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


breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e095a), internal=True
)
# Keep this as an ordinary conditional breakpoint with no Python stop method.
# GDB can then reject earlier ray intervals without crossing into Python.
breakpoint.condition = (
    "(*(unsigned long long*)$rbx == %d) && "
    "(*(unsigned long long*)($rbx + 0x50) >= %d) && "
    "(*(unsigned long long*)($rbx + 0x50) <= %d)"
) % (RAW_VPTR, START_NS, END_NS)


class PredictorCorrectionEntry(gdb.Breakpoint):
    def __init__(self, gravity_breakpoint):
        super().__init__("*%#x" % (IMAGE_BASE + 0x4CA2D0), internal=True)
        self.gravity_breakpoint = gravity_breakpoint

    def stop(self):
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            timestamp_ns = struct.unpack(
                "<Q",
                bytes(gdb.selected_inferior().read_memory(timestamp_pointer, 8)),
            )[0]
        except gdb.MemoryError:
            return False
        if timestamp_ns < ACTIVATE_NS:
            return False
        self.gravity_breakpoint.enabled = True
        self.enabled = False
        gdb.write(
            "enabled raw IMU correction-window probe at %d ns\n"
            % timestamp_ns
        )
        return False


if ACTIVATE_NS:
    breakpoint.enabled = False
    PredictorCorrectionEntry(breakpoint)
end
continue
python
owner = int(gdb.parse_and_eval("$rbx"))
inferior = gdb.selected_inferior()
current_time, initial_time = struct.unpack(
    "<QQ", bytes(inferior.read_memory(owner + 0x50, 16))
)
quaternion = struct.unpack(
    "<4d", bytes(inferior.read_memory(owner + 0xb0, 32))
)
gravity = struct.unpack(
    "<3d", bytes(inferior.read_memory(owner + 0xd0, 24))
)
OUTPUT.write_text(json.dumps({
    "stage": "after_correction",
    "current_time_ns": current_time,
    "initial_time_ns": initial_time,
    "quaternion_xyzw": quaternion,
    "gravity": gravity,
}, indent=2) + "\n")
gdb.write("captured raw IMU correction window\n")
end
quit
