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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_pose_predictor_corrections.json"
))
TARGET_NS = int(os.environ.get("NAVVIS_PROBE_TARGET_NS", "0"))


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


BASE = image_base("surveyorslam_processing_node")
records = []


def flush():
    OUTPUT.write_text(json.dumps(records, indent=2) + "\n")


class CorrectionBreakpoint(gdb.Breakpoint):
    def __init__(self, offset, name, shifted_abi=False):
        super().__init__("*%#x" % (BASE + offset), internal=True)
        self.offset = offset
        self.name = name
        self.shifted_abi = shifted_abi

    def stop(self):
        inferior = gdb.selected_inferior()
        owner_register = "$rsi" if self.shifted_abi else "$rdi"
        timestamp_register = "$rdx" if self.shifted_abi else "$rsi"
        owner = int(gdb.parse_and_eval(owner_register))
        timestamp_pointer = int(gdb.parse_and_eval(timestamp_register))
        try:
            vptr = struct.unpack(
                "<Q", bytes(inferior.read_memory(owner, 8))
            )[0]
            timestamp_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(timestamp_pointer, 8))
            )[0]
        except gdb.MemoryError:
            return False
        records.append({
            "method": self.name,
            "method_offset": self.offset,
            "owner": owner,
            "vptr_offset": vptr - BASE,
            "timestamp_ns": timestamp_ns,
        })
        if len(records) <= 32 or len(records) % 128 == 0:
            flush()
        if TARGET_NS and timestamp_ns >= TARGET_NS:
            flush()
            gdb.write(
                "captured correction boundary at %d ns via %s\n"
                % (timestamp_ns, self.name)
            )
            gdb.execute("quit")
        return False


CorrectionBreakpoint(0x4C8D30, "const_vel_init_wrapper")
CorrectionBreakpoint(0x4C9500, "constant_velocity")
CorrectionBreakpoint(0x4CBF40, "imu_trajectory")
CorrectionBreakpoint(0x4C8F00, "const_vel_init_wrapper_correction", True)
CorrectionBreakpoint(0x4CA2D0, "constant_velocity_correction", True)
CorrectionBreakpoint(0x4CC990, "imu_trajectory_correction", True)
end
continue
