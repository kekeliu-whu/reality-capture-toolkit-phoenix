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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_tracker_state.json"
))
CURRENT_NS = int(os.environ["NAVVIS_PROBE_CURRENT_NS"])
AT_OR_AFTER = os.environ.get("NAVVIS_PROBE_AT_OR_AFTER", "0") != "0"
ANY_VPTR = os.environ.get("NAVVIS_PROBE_ANY_VPTR", "0") != "0"


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
    "*%#x" % (IMAGE_BASE + 0x6e1148), internal=True
)
time_predicate = ">=" if AT_OR_AFTER else "=="
if ANY_VPTR:
    breakpoint.condition = (
        "*(unsigned long long*)($rbx + 0x50) %s %d"
    ) % (time_predicate, CURRENT_NS)
else:
    breakpoint.condition = (
        "(*(unsigned long long*)$rbx == %d) && "
        "(*(unsigned long long*)($rbx + 0x50) %s %d)"
    ) % (RAW_VPTR, time_predicate, CURRENT_NS)
end
continue
python
owner = int(gdb.parse_and_eval("$rbx"))
inferior = gdb.selected_inferior()
vptr = struct.unpack("<Q", bytes(inferior.read_memory(owner, 8)))[0]
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
    "stage": "before_interval",
    "vptr": vptr,
    "vptr_offset": vptr - IMAGE_BASE,
    "current_time_ns": current_time,
    "initial_time_ns": initial_time,
    "quaternion_xyzw": quaternion,
    "gravity": gravity,
}, indent=2) + "\n")
gdb.write("captured raw IMU tracker state\n")
end
quit
