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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_interval_stages.json"
))
CURRENT_NS = int(os.environ["NAVVIS_PROBE_CURRENT_NS"])


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


def doubles(address, count):
    return struct.unpack(
        "<%dd" % count,
        bytes(gdb.selected_inferior().read_memory(address, 8 * count)),
    )


def state(owner):
    return {
        "owner": owner,
        "current_time_ns": struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(owner + 0x50, 8))
        )[0],
        "quaternion_xyzw": doubles(owner + 0xb0, 4),
        "gravity": doubles(owner + 0xd0, 3),
    }


IMAGE_BASE = image_base("surveyorslam_processing_node")
RAW_VPTR = IMAGE_BASE + 0x1830660
condition = (
    "(*(unsigned long long*)$rbx == %d) && "
    "(*(unsigned long long*)($rbx + 0x50) == %d)"
) % (RAW_VPTR, CURRENT_NS)

angular_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e1554), internal=True
)
angular_breakpoint.condition = condition
rotation_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e16cf), internal=True
)
rotation_breakpoint.enabled = False
gravity_entry_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e0d60), internal=True
)
gravity_entry_breakpoint.enabled = False
helper_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e0280), internal=True
)
helper_breakpoint.enabled = False
blend_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e06a7), internal=True
)
blend_breakpoint.enabled = False
correction_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e07f5), internal=True
)
correction_breakpoint.enabled = False
finish_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e095a), internal=True
)
finish_breakpoint.enabled = False
records = []
end
continue
python
owner = int(gdb.parse_and_eval("$rbx"))
records.append({"stage": "after_angular_orientation", **state(owner)})
angular_breakpoint.enabled = False
rotation_breakpoint.enabled = True
end
continue
python
records.append({"stage": "after_gravity_rotation", **state(owner)})
rotation_breakpoint.enabled = False
gravity_entry_breakpoint.enabled = True
end
continue
python
options = int(gdb.parse_and_eval("$rdi"))
sample = int(gdb.parse_and_eval("$rsi"))
records.append({
    "stage": "gravity_update_entry",
    "dt_s": float(gdb.parse_and_eval("$xmm0.v2_double[0]")),
    "elapsed_s": float(gdb.parse_and_eval("$xmm1.v2_double[0]")),
    "acceleration": doubles(sample, 3),
    "time_constants": doubles(options + 0x20, 5),
})
gravity_entry_breakpoint.enabled = False
helper_breakpoint.enabled = True
end
continue
python
helper_owner = int(gdb.parse_and_eval("$rdi"))
sample = int(gdb.parse_and_eval("$rsi"))
records.append({
    "stage": "helper_entry",
    **state(helper_owner),
    "acceleration": doubles(sample, 3),
    "alpha": float(gdb.parse_and_eval("$xmm0.v2_double[0]")),
})
helper_breakpoint.enabled = False
blend_breakpoint.enabled = True
end
continue
python
records.append({"stage": "after_gravity_blend", **state(owner)})
blend_breakpoint.enabled = False
correction_breakpoint.enabled = True
end
continue
python
stack = int(gdb.parse_and_eval("$rsp"))
records.append({
    "stage": "gravity_correction",
    **state(owner),
    "correction_xyzw": doubles(stack + 0x70, 4),
    "expected_gravity": doubles(stack + 0x50, 3),
})
correction_breakpoint.enabled = False
finish_breakpoint.enabled = True
end
continue
python
records.append({"stage": "after_correction", **state(owner)})
OUTPUT.write_text(json.dumps(records, indent=2) + "\n")
gdb.write("captured raw IMU interval stages\n")
end
quit
