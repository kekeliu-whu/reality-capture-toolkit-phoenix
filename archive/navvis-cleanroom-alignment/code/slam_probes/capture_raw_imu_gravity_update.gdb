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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_raw_imu_gravity_update.json"
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


def unpack_state(raw):
    return {
        "current_time_ns": struct.unpack_from("<Q", raw, 0x50)[0],
        "initial_time_ns": struct.unpack_from("<Q", raw, 0x58)[0],
        "quaternion_xyzw": struct.unpack_from("<4d", raw, 0xB0),
        "gravity": struct.unpack_from("<3d", raw, 0xD0),
        "raw_hex": raw.hex(),
    }


class GravityUpdateReturn(gdb.FinishBreakpoint):
    def __init__(self, tracker, acceleration, alpha, before):
        super().__init__(internal=True)
        self.tracker = tracker
        self.acceleration = acceleration
        self.alpha = alpha
        self.before = before

    def stop(self):
        inferior = gdb.selected_inferior()
        after = bytes(inferior.read_memory(self.tracker, 0xF0))
        record = {
            "alpha": self.alpha,
            "acceleration": self.acceleration,
            "before": unpack_state(self.before),
            "after": unpack_state(after),
        }
        OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
        gdb.write(
            "captured raw-IMU gravity update from %d ns\n"
            % record["before"]["current_time_ns"]
        )
        gdb.execute("quit")
        return False


class GravityUpdateEntry(gdb.Breakpoint):
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
        acceleration_pointer = int(gdb.parse_and_eval("$rsi"))
        try:
            tracker_vptr = struct.unpack(
                "<Q", bytes(inferior.read_memory(tracker, 8))
            )[0]
            before = bytes(inferior.read_memory(tracker, 0xF0))
            current_time_ns = struct.unpack_from("<Q", before, 0x50)[0]
            acceleration = struct.unpack(
                "<3d", bytes(inferior.read_memory(acceleration_pointer, 24))
            )
            alpha = float(gdb.parse_and_eval("$xmm0.v2_double[0]"))
        except (gdb.MemoryError, gdb.error):
            return False
        if tracker_vptr != RAW_TRACKER_VPTR or current_time_ns != CURRENT_TIME_NS:
            return False
        GravityUpdateReturn(tracker, acceleration, alpha, before)
        return False


class CorrectionEntry(gdb.Breakpoint):
    def __init__(self, gravity_update_entry):
        super().__init__("*%#x" % (BASE + 0x4CA2D0), internal=True)
        self.gravity_update_entry = gravity_update_entry

    def stop(self):
        inferior = gdb.selected_inferior()
        predictor = int(gdb.parse_and_eval("$rsi"))
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            timestamp_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(timestamp_pointer, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if timestamp_ns < ACTIVATE_NS:
            return False
        self.gravity_update_entry.enabled = True
        self.enabled = False
        gdb.write(
            "enabled gravity-update probe after correction at %d ns\n"
            % timestamp_ns
        )
        return False


gravity_update_entry = GravityUpdateEntry()
if ACTIVATE_NS:
    gravity_update_entry.enabled = False
    CorrectionEntry(gravity_update_entry)
end
continue
