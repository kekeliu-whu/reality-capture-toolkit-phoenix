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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_constant_velocity_prediction_stages.json"
))
TARGET_NS = int(os.environ["NAVVIS_PROBE_TARGET_NS"])
ACTIVATE_NS = int(os.environ.get("NAVVIS_PROBE_ACTIVATE_NS", "0"))
CALLER_RETURN_OFFSET = int(
    os.environ.get("NAVVIS_PROBE_CALLER_RETURN_OFFSET", "0"), 0
)
CALL_SITE_OFFSET = int(
    os.environ.get("NAVVIS_PROBE_CALL_SITE_OFFSET", "0"), 0
)


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


BASE = image_base("surveyorslam_processing_node")
CONSTANT_VELOCITY_VPTR = BASE + 0x1827BA0
FINAL_COMPOSE_RETURN = BASE + 0x4C9D57
active_thread = None
active_entry_rsp = None
active_record = None
active_predictor = None


def pose_bytes(address):
    raw = bytes(gdb.selected_inferior().read_memory(address, 0x40))
    return {
        "translation": struct.unpack_from("<3d", raw, 0),
        "quaternion_xyzw": struct.unpack_from("<4d", raw, 0x20),
        "raw_hex": raw.hex(),
    }


def quaternion_bytes(address):
    raw = bytes(gdb.selected_inferior().read_memory(address, 0x20))
    return {
        "quaternion_xyzw": struct.unpack("<4d", raw),
        "raw_hex": raw.hex(),
    }


class ComposeReturn(gdb.FinishBreakpoint):
    def __init__(self, output):
        super().__init__(internal=True)
        self.output = output

    def stop(self):
        active_record["compose_output"] = pose_bytes(self.output)
        return False


class PredictionReturn(gdb.FinishBreakpoint):
    def __init__(self, output, predictor):
        super().__init__(internal=True)
        self.output = output
        self.predictor = predictor

    def stop(self):
        active_record["prediction_output"] = pose_bytes(self.output)
        active_record["predictor_raw_hex"] = bytes(
            gdb.selected_inferior().read_memory(self.predictor, 0x268)
        ).hex()
        OUTPUT.write_text(json.dumps(active_record, indent=2) + "\n")
        gdb.write("captured constant-velocity prediction and compose stages\n")
        gdb.execute("quit")
        return False


class FinalCompose(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x32B650), internal=True)
        self.enabled = False

    def stop(self):
        if active_thread is None or gdb.selected_thread().num != active_thread:
            return False
        stack_pointer = int(gdb.parse_and_eval("$rsp"))
        if active_entry_rsp - stack_pointer != 0x180:
            return False
        return_address = struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(
                stack_pointer, 8
            ))
        )[0]
        if return_address != FINAL_COMPOSE_RETURN:
            return False
        output = int(gdb.parse_and_eval("$rdi"))
        anchor = int(gdb.parse_and_eval("$rsi"))
        increment = int(gdb.parse_and_eval("$rdx"))
        active_record.update({
            "timestamp_ns": TARGET_NS,
            "anchor": pose_bytes(anchor),
            "increment": pose_bytes(increment),
        })
        ComposeReturn(output)
        self.enabled = False
        return False


class TrackerOrientationReturn(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x4C9E7C), internal=True)
        self.enabled = False

    def stop(self):
        if active_thread is None or gdb.selected_thread().num != active_thread:
            return False
        stack_pointer = int(gdb.parse_and_eval("$rsp"))
        if active_entry_rsp - stack_pointer != 0x178:
            return False
        active_record["anchor_tracker"] = quaternion_bytes(
            active_predictor + 0x1B0
        )
        active_record["tracker"] = quaternion_bytes(stack_pointer + 0xD0)
        self.enabled = False
        return False


class PredictionEntry(gdb.Breakpoint):
    def __init__(self, compose_breakpoint):
        super().__init__("*%#x" % (BASE + 0x4C97C0), internal=True)
        self.compose_breakpoint = compose_breakpoint
        self.condition = (
            "(*(unsigned long long*)$rsi == %d) && "
            "(*(unsigned long long*)$rdx == %d)"
            % (CONSTANT_VELOCITY_VPTR, TARGET_NS)
        )

    def stop(self):
        global active_thread, active_entry_rsp, active_record, active_predictor
        entry_rsp = int(gdb.parse_and_eval("$rsp"))
        return_address = struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(entry_rsp, 8))
        )[0]
        if (
            not CALL_SITE_OFFSET
            and
            CALLER_RETURN_OFFSET
            and return_address != BASE + CALLER_RETURN_OFFSET
        ):
            return False
        active_thread = gdb.selected_thread().num
        active_entry_rsp = entry_rsp
        output = int(gdb.parse_and_eval("$rdi"))
        predictor = int(gdb.parse_and_eval("$rsi"))
        active_predictor = predictor
        active_record = {
            "timestamp_ns": TARGET_NS,
            "caller_return_offset": return_address - BASE,
        }
        PredictionReturn(output, predictor)
        self.compose_breakpoint.enabled = True
        tracker_breakpoint.enabled = True
        self.enabled = False
        return False


class PredictionCallSite(gdb.Breakpoint):
    def __init__(self, prediction_breakpoint):
        super().__init__("*%#x" % (BASE + CALL_SITE_OFFSET), internal=True)
        self.prediction_breakpoint = prediction_breakpoint

    def stop(self):
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            timestamp_ns = struct.unpack(
                "<Q",
                bytes(gdb.selected_inferior().read_memory(timestamp_pointer, 8)),
            )[0]
        except gdb.MemoryError:
            return False
        if timestamp_ns != TARGET_NS:
            return False
        self.prediction_breakpoint.enabled = True
        self.enabled = False
        return False


class CorrectionEntry(gdb.Breakpoint):
    def __init__(self, prediction_breakpoint):
        super().__init__("*%#x" % (BASE + 0x4CA2D0), internal=True)
        self.prediction_breakpoint = prediction_breakpoint

    def stop(self):
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        timestamp_ns = struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(timestamp_pointer, 8))
        )[0]
        if timestamp_ns < ACTIVATE_NS:
            return False
        self.prediction_breakpoint.enabled = True
        self.enabled = False
        return False


compose_breakpoint = FinalCompose()
tracker_breakpoint = TrackerOrientationReturn()
prediction_breakpoint = PredictionEntry(compose_breakpoint)
if CALL_SITE_OFFSET:
    prediction_breakpoint.enabled = False
    call_site_breakpoint = PredictionCallSite(prediction_breakpoint)
    if ACTIVATE_NS:
        call_site_breakpoint.enabled = False
        CorrectionEntry(call_site_breakpoint)
elif ACTIVATE_NS:
    prediction_breakpoint.enabled = False
    CorrectionEntry(prediction_breakpoint)
end
continue
