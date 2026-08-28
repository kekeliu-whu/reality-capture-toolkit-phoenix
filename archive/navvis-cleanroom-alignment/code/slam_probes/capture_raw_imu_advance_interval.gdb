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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_raw_imu_advance_interval.json"
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


class AdvanceInterval(gdb.Breakpoint):
    def __init__(self):
        # RawImuTracker::Advance immediately before the gravity-update wrapper.
        # r13 is the selected interval end; the outer request is saved at
        # rsp+0x68 and the separately rounded interval duration is at rsp.
        super().__init__("*%#x" % (BASE + 0x6E17CB), internal=True)
        self.condition = (
            "(*(unsigned long long*)$rbx == %d) && "
            "(*(unsigned long long*)($rbx + 0x50) == %d)"
            % (RAW_TRACKER_VPTR, CURRENT_TIME_NS)
        )

    def stop(self):
        inferior = gdb.selected_inferior()
        tracker = int(gdb.parse_and_eval("$rbx"))
        stack = int(gdb.parse_and_eval("$rsp"))
        try:
            current_time_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(tracker + 0x50, 8))
            )[0]
            requested_time_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(stack + 0x68, 8))
            )[0]
            interval_end_ns = int(gdb.parse_and_eval("$r13"))
            dt_s = struct.unpack(
                "<d", bytes(inferior.read_memory(stack, 8))
            )[0]
            acceleration = struct.unpack(
                "<3d", bytes(inferior.read_memory(stack + 0xD0, 24))
            )
        except (gdb.MemoryError, gdb.error):
            return False
        if current_time_ns != CURRENT_TIME_NS:
            return False
        OUTPUT.write_text(json.dumps({
            "current_time_ns": current_time_ns,
            "interval_end_ns": interval_end_ns,
            "requested_time_ns": requested_time_ns,
            "dt_s": dt_s,
            "acceleration": acceleration,
        }, indent=2) + "\n")
        gdb.write(
            "captured raw IMU interval %d -> %d (request %d)\n"
            % (current_time_ns, interval_end_ns, requested_time_ns)
        )
        gdb.execute("quit")
        return False


class PredictorCorrectionEntry(gdb.Breakpoint):
    def __init__(self, interval_breakpoint):
        super().__init__("*%#x" % (BASE + 0x4CA2D0), internal=True)
        self.interval_breakpoint = interval_breakpoint

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
        self.interval_breakpoint.enabled = True
        self.enabled = False
        gdb.write(
            "enabled raw IMU interval probe at %d ns\n" % timestamp_ns
        )
        return False


interval_breakpoint = AdvanceInterval()
if ACTIVATE_NS:
    interval_breakpoint.enabled = False
    PredictorCorrectionEntry(interval_breakpoint)
end
continue
