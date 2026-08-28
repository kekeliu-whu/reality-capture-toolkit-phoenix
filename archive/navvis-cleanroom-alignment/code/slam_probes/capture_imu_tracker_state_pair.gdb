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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_tracker_state_pair.json"
))
REFERENCE_NS = int(os.environ["NAVVIS_PROBE_REFERENCE_NS"])
TARGET_NS = int(os.environ["NAVVIS_PROBE_TARGET_NS"])


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


def tracker_state(owner):
    inferior = gdb.selected_inferior()
    current_time, initial_time = struct.unpack(
        "<QQ", bytes(inferior.read_memory(owner + 0x50, 16))
    )
    return {
        "owner": owner,
        "current_time_ns": current_time,
        "initial_time_ns": initial_time,
        "quaternion_xyzw": struct.unpack(
            "<4d", bytes(inferior.read_memory(owner + 0xb0, 32))
        ),
        "gravity": struct.unpack(
            "<3d", bytes(inferior.read_memory(owner + 0xd0, 24))
        ),
    }


IMAGE_BASE = image_base("surveyorslam_processing_node")
RAW_VPTR = IMAGE_BASE + 0x1830660
reference_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e1148), internal=True
)
reference_breakpoint.condition = (
    "(*(unsigned long long*)$rbx == %d) && "
    "(*(unsigned long long*)($rbx + 0x50) == %d)"
) % (RAW_VPTR, REFERENCE_NS)
records = {}
end
continue
python
owner = int(gdb.parse_and_eval("$rbx"))
records["reference"] = tracker_state(owner)
reference_breakpoint.enabled = False
target_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e1148), internal=True
)
target_breakpoint.condition = (
    "($rbx == %d) && "
    "(*(unsigned long long*)($rbx + 0x50) == %d)"
) % (owner, TARGET_NS)
end
continue
python
records["target"] = tracker_state(int(gdb.parse_and_eval("$rbx")))
OUTPUT.write_text(json.dumps(records, indent=2) + "\n")
gdb.write("captured raw IMU tracker state pair\n")
end
quit
