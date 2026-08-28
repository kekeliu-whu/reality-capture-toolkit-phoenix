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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_vendor_imu_tracker_steps.json"
))
STEP_LIMIT = int(os.environ.get("NAVVIS_PROBE_STEP_LIMIT", "32"))
START_NS = int(os.environ.get("NAVVIS_PROBE_START_NS", "0"))
END_NS = int(os.environ.get("NAVVIS_PROBE_END_NS", str((1 << 63) - 1)))
INITIAL_TIME_NS = int(os.environ.get("NAVVIS_PROBE_INITIAL_TIME_NS", "0"))


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
records = []
captured_corrections = 0


def finish(message):
    OUTPUT.write_text(json.dumps(records, indent=2) + "\n")
    gdb.write(message + "\n")
    gdb.execute("quit")


def state(owner, stage, extra=None):
    inferior = gdb.selected_inferior()
    vptr = struct.unpack("<Q", bytes(inferior.read_memory(owner, 8)))[0]
    if vptr != RAW_VPTR:
        return
    current_time, initial_time = struct.unpack(
        "<QQ", bytes(inferior.read_memory(owner + 0x50, 16))
    )
    if current_time < START_NS or current_time > END_NS:
        return False
    quaternion = struct.unpack(
        "<4d", bytes(inferior.read_memory(owner + 0xb0, 32))
    )
    gravity = struct.unpack(
        "<3d", bytes(inferior.read_memory(owner + 0xd0, 24))
    )
    record = {
        "stage": stage,
        "owner": owner,
        "current_time_ns": current_time,
        "initial_time_ns": initial_time,
        "quaternion_xyzw": quaternion,
        "gravity": gravity,
    }
    if extra:
        record.update(extra)
    records.append(record)
    return True


class GravityUpdateEntryBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x6e0d60), internal=True)

    def stop(self):
        options = int(gdb.parse_and_eval("$rdi"))
        sample = int(gdb.parse_and_eval("$rsi"))
        inferior = gdb.selected_inferior()
        dt_s = float(gdb.parse_and_eval("$xmm0.v2_double[0]"))
        elapsed_s = float(gdb.parse_and_eval("$xmm1.v2_double[0]"))
        endpoint_ns = INITIAL_TIME_NS + int(round(elapsed_s * 1.0e9))
        if START_NS <= endpoint_ns <= END_NS:
            acceleration = struct.unpack(
                "<3d", bytes(inferior.read_memory(sample, 24))
            )
            constants = struct.unpack(
                "<5d", bytes(inferior.read_memory(options + 0x20, 40))
            )
            records.append({
                "stage": "gravity_update_entry",
                "elapsed_s": elapsed_s,
                "endpoint_ns": endpoint_ns,
                "dt_s": dt_s,
                "acceleration": acceleration,
                "initial_time_constant_s": constants[0],
                "steady_time_constant_s": constants[1],
                "max_gravity_norm_error_mps2": constants[2],
                "init_duration_s": constants[3],
                "fade_duration_s": constants[4],
            })
        return False


class HelperEntryBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x6e0280), internal=True)

    def stop(self):
        owner = int(gdb.parse_and_eval("$rdi"))
        sample = int(gdb.parse_and_eval("$rsi"))
        inferior = gdb.selected_inferior()
        acceleration = struct.unpack(
            "<3d", bytes(inferior.read_memory(sample, 24))
        )
        alpha = float(gdb.parse_and_eval("$xmm0.v2_double[0]"))
        state(owner, "helper_entry", {
            "acceleration": acceleration,
            "alpha": alpha,
        })
        return False


class StateBreakpoint(gdb.Breakpoint):
    def __init__(self, offset, stage):
        super().__init__("*%#x" % (IMAGE_BASE + offset), internal=True)
        self.stage = stage

    def stop(self):
        global captured_corrections
        owner = int(gdb.parse_and_eval("$rbx"))
        captured = state(owner, self.stage)
        if self.stage == "after_correction" and captured:
            captured_corrections += 1
            if captured_corrections >= STEP_LIMIT:
                finish("captured %d raw IMU update steps" % STEP_LIMIT)
        elif self.stage == "after_correction":
            inferior = gdb.selected_inferior()
            current_time = struct.unpack(
                "<Q", bytes(inferior.read_memory(owner + 0x50, 8))
            )[0]
            if current_time > END_NS and records:
                finish("captured raw IMU update window")
        return False


class GravityCorrectionBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x6e07f5), internal=True)

    def stop(self):
        owner = int(gdb.parse_and_eval("$rbx"))
        stack = int(gdb.parse_and_eval("$rsp"))
        correction = struct.unpack(
            "<4d", bytes(gdb.selected_inferior().read_memory(stack + 0x70, 32))
        )
        expected_gravity = struct.unpack(
            "<3d", bytes(gdb.selected_inferior().read_memory(stack + 0x50, 24))
        )
        state(owner, "gravity_correction", {
            "correction_xyzw": correction,
            "expected_gravity": expected_gravity,
        })
        return False


# Each address is the instruction immediately after the corresponding stores.
GravityUpdateEntryBreakpoint()
HelperEntryBreakpoint()
StateBreakpoint(0x6e1554, "after_angular_orientation")
StateBreakpoint(0x6e16cf, "after_gravity_rotation")
StateBreakpoint(0x6e06a7, "after_gravity_blend")
GravityCorrectionBreakpoint()
StateBreakpoint(0x6e095a, "after_correction")
end
continue
