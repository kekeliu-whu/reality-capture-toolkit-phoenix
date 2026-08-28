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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_angular_operands.json"
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


def xmm(name):
    return tuple(float(gdb.parse_and_eval("$%s.v2_double[%d]" % (name, lane)))
                 for lane in range(2))


IMAGE_BASE = image_base("surveyorslam_processing_node")
RAW_VPTR = IMAGE_BASE + 0x1830660
condition = (
    "(*(unsigned long long*)$rbx == %d) && "
    "(*(unsigned long long*)($rbx + 0x50) == %d)"
) % (RAW_VPTR, CURRENT_NS)

rotation_vector_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e1421), internal=True
)
rotation_vector_breakpoint.condition = condition
delta_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e148f), internal=True
)
delta_breakpoint.condition = condition
delta_breakpoint.enabled = False
inverse_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e15ab), internal=True
)
inverse_breakpoint.condition = condition
inverse_breakpoint.enabled = False
rotation_breakpoint = gdb.Breakpoint(
    "*%#x" % (IMAGE_BASE + 0x6e16b9), internal=True
)
rotation_breakpoint.condition = condition
rotation_breakpoint.enabled = False
records = {}
end
continue
python
owner = int(gdb.parse_and_eval("$rbx"))
stack = int(gdb.parse_and_eval("$rsp"))
rotation_xy = xmm("xmm2")
records["rotation_vector"] = {
    "current_time_ns": CURRENT_NS,
    "rotation_vector": rotation_xy + (float(
        gdb.parse_and_eval("$xmm1.v2_double[0]")
    ),),
    "duration_s": float(gdb.parse_and_eval("$xmm8.v2_double[0]")),
}
rotation_vector_breakpoint.enabled = False
delta_breakpoint.enabled = True
end
continue
python
owner = int(gdb.parse_and_eval("$rbx"))
stack = int(gdb.parse_and_eval("$rsp"))
delta_xy = xmm("xmm2")
delta_zw = doubles(stack + 0xc0, 2)
records["before_product"] = {
    "current_time_ns": CURRENT_NS,
    "orientation_xyzw": doubles(owner + 0xb0, 4),
    "gravity": doubles(owner + 0xd0, 3),
    "delta_xyzw": delta_xy + delta_zw,
}
delta_breakpoint.enabled = False
inverse_breakpoint.enabled = True
end
continue
python
owner = int(gdb.parse_and_eval("$rbx"))
stack = int(gdb.parse_and_eval("$rsp"))
records["before_rotation"] = {
    "orientation_xyzw": doubles(owner + 0xb0, 4),
    "gravity": doubles(owner + 0xd0, 3),
    "inverse_delta_xyzw": doubles(stack + 0xd0, 4),
}
inverse_breakpoint.enabled = False
rotation_breakpoint.enabled = True
end
continue
python
stack = int(gdb.parse_and_eval("$rsp"))
records["rotation_result"] = {
    "gravity": doubles(stack + 0x90, 3),
}
OUTPUT.write_text(json.dumps(records, indent=2) + "\n")
gdb.write("captured raw IMU angular operands\n")
end
quit
