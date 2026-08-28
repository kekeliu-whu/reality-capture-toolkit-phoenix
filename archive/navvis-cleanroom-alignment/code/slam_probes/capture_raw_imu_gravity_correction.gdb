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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_raw_imu_gravity_correction.json"
))
CURRENT_TIME_NS = int(os.environ["NAVVIS_PROBE_CURRENT_TIME_NS"])
ACTIVATE_NS = int(os.environ.get("NAVVIS_PROBE_ACTIVATE_NS", "0"))


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


BASE = image_base("surveyorslam_processing_node")
RAW_TRACKER_VPTR = BASE + 0x1830660


class CorrectionReady(gdb.Breakpoint):
    def __init__(self, tracker):
        super().__init__("*%#x" % (BASE + 0x6E07F5), internal=True)
        self.tracker = tracker

    def stop(self):
        inferior = gdb.selected_inferior()
        tracker = int(gdb.parse_and_eval("$rbx"))
        if tracker != self.tracker:
            return False
        stack = int(gdb.parse_and_eval("$rsp"))
        tracker_raw = bytes(inferior.read_memory(tracker, 0xF0))
        correction_raw = bytes(inferior.read_memory(stack + 0x70, 32))
        record = {
            "current_time_ns": struct.unpack_from("<Q", tracker_raw, 0x50)[0],
            "orientation_xyzw": struct.unpack_from("<4d", tracker_raw, 0xB0),
            "updated_gravity": struct.unpack_from("<3d", tracker_raw, 0xD0),
            "correction_xyzw": struct.unpack("<4d", correction_raw),
            "correction_raw_hex": correction_raw.hex(),
        }
        OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
        gdb.write(
            "captured raw-IMU gravity correction at %d ns\n"
            % record["current_time_ns"]
        )
        gdb.execute("quit")
        return False


class GravityKernelEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x6E0280), internal=True)
        self.condition = (
            "(*(unsigned long long*)$rdi == %d) && "
            "(*(unsigned long long*)($rdi + 0x50) == %d)"
            % (RAW_TRACKER_VPTR, CURRENT_TIME_NS)
        )

    def stop(self):
        inferior = gdb.selected_inferior()
        tracker = int(gdb.parse_and_eval("$rdi"))
        try:
            tracker_vptr = struct.unpack(
                "<Q", bytes(inferior.read_memory(tracker, 8))
            )[0]
            current_time_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(tracker + 0x50, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if tracker_vptr != RAW_TRACKER_VPTR or current_time_ns != CURRENT_TIME_NS:
            return False
        self.enabled = False
        CorrectionReady(tracker)
        return False


class PredictorCorrectionEntry(gdb.Breakpoint):
    def __init__(self, gravity_kernel_entry):
        super().__init__("*%#x" % (BASE + 0x4CA2D0), internal=True)
        self.gravity_kernel_entry = gravity_kernel_entry

    def stop(self):
        inferior = gdb.selected_inferior()
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            timestamp_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(timestamp_pointer, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if timestamp_ns < ACTIVATE_NS:
            return False
        self.gravity_kernel_entry.enabled = True
        self.enabled = False
        gdb.write(
            "enabled gravity-correction probe after pose correction at %d ns\n"
            % timestamp_ns
        )
        return False


gravity_kernel_entry = GravityKernelEntry()
if ACTIVATE_NS:
    gravity_kernel_entry.enabled = False
    PredictorCorrectionEntry(gravity_kernel_entry)
end
continue
