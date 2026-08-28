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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_constant_velocity_prediction.json"
))
TARGET_NS = int(os.environ.get("NAVVIS_PROBE_TARGET_NS", "0"))
TARGETS_NS = {
    int(value)
    for value in os.environ.get("NAVVIS_PROBE_TARGETS_NS", "").split(",")
    if value.strip()
}
if TARGET_NS:
    TARGETS_NS.add(TARGET_NS)
if not TARGETS_NS:
    raise RuntimeError(
        "set NAVVIS_PROBE_TARGET_NS or NAVVIS_PROBE_TARGETS_NS"
    )
AT_OR_AFTER = os.environ.get("NAVVIS_PROBE_AT_OR_AFTER", "0") != "0"
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
CONSTANT_VELOCITY_VPTR = BASE + 0x1827BA0
records = []
captured_timestamps = set()


class PredictionReturn(gdb.FinishBreakpoint):
    def __init__(self, output, predictor, timestamp_ns):
        super().__init__(internal=True)
        self.output = output
        self.predictor = predictor
        self.timestamp_ns = timestamp_ns

    def stop(self):
        inferior = gdb.selected_inferior()
        output_bytes = bytes(inferior.read_memory(self.output, 0x48))
        translation = struct.unpack_from("<3d", output_bytes, 0)
        quaternion = struct.unpack_from("<4d", output_bytes, 0x20)
        valid = output_bytes[0x40] != 0
        predictor_bytes = bytes(inferior.read_memory(self.predictor, 0x268))
        tracker = struct.unpack_from("<Q", predictor_bytes, 0x38)[0]
        tracker_vptr = 0
        tracker_bytes = b""
        if tracker:
            tracker_bytes = bytes(inferior.read_memory(tracker, 0x100))
            tracker_vptr = struct.unpack_from("<Q", tracker_bytes, 0)[0]
        record = {
            "timestamp_ns": self.timestamp_ns,
            "prediction": {
                "translation": translation,
                "quaternion_xyzw": quaternion,
                "valid": valid,
                "raw_hex": output_bytes.hex(),
            },
            "predictor": {
                "address": self.predictor,
                "vptr_offset": struct.unpack_from("<Q", predictor_bytes, 0)[0] - BASE,
                "current_time_ns": struct.unpack_from("<Q", predictor_bytes, 0x248)[0],
                "raw_hex": predictor_bytes.hex(),
            },
            "tracker": {
                "address": tracker,
                "vptr_offset": tracker_vptr - BASE if tracker_vptr else 0,
                "raw_hex": tracker_bytes.hex(),
            },
        }
        records.append(record)
        captured_timestamps.add(self.timestamp_ns)
        OUTPUT.write_text(json.dumps(records, indent=2) + "\n")
        gdb.write(
            "captured constant-velocity prediction at %d ns, tracker vptr %#x\n"
            % (self.timestamp_ns, record["tracker"]["vptr_offset"])
        )
        if captured_timestamps == TARGETS_NS:
            gdb.execute("quit")
        return False


class PredictionEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x4C97C0), internal=True)
        if AT_OR_AFTER:
            target_condition = "(*(unsigned long long*)$rdx >= %d)" % next(
                iter(TARGETS_NS)
            )
        else:
            target_condition = "(" + " || ".join(
                "(*(unsigned long long*)$rdx == %d)" % timestamp_ns
                for timestamp_ns in sorted(TARGETS_NS)
            ) + ")"
        self.condition = (
            "(*(unsigned long long*)$rsi == %d) && %s"
            % (CONSTANT_VELOCITY_VPTR, target_condition)
        )

    def stop(self):
        inferior = gdb.selected_inferior()
        output = int(gdb.parse_and_eval("$rdi"))
        predictor = int(gdb.parse_and_eval("$rsi"))
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            predictor_vptr = struct.unpack(
                "<Q", bytes(inferior.read_memory(predictor, 8))
            )[0]
            timestamp_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(timestamp_pointer, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if predictor_vptr != CONSTANT_VELOCITY_VPTR:
            return False
        if timestamp_ns in captured_timestamps:
            return False
        if AT_OR_AFTER:
            if len(TARGETS_NS) != 1 or timestamp_ns < next(iter(TARGETS_NS)):
                return False
        elif timestamp_ns not in TARGETS_NS:
            return False
        PredictionReturn(output, predictor, timestamp_ns)
        return False


class CorrectionEntry(gdb.Breakpoint):
    def __init__(self, prediction_entry):
        # Virtual slot 7 is the per-node scan-matching correction.  Its hidden
        # return object shifts `this` and the timestamp pointer to rsi/rdx.
        super().__init__("*%#x" % (BASE + 0x4CA2D0), internal=True)
        self.prediction_entry = prediction_entry

    def stop(self):
        inferior = gdb.selected_inferior()
        predictor = int(gdb.parse_and_eval("$rsi"))
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            predictor_vptr = struct.unpack(
                "<Q", bytes(inferior.read_memory(predictor, 8))
            )[0]
            timestamp_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(timestamp_pointer, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if (
            predictor_vptr != CONSTANT_VELOCITY_VPTR
            or timestamp_ns < ACTIVATE_NS
        ):
            return False
        self.prediction_entry.enabled = True
        self.enabled = False
        gdb.write(
            "enabled point prediction probe after correction at %d ns\n"
            % timestamp_ns
        )
        return False


prediction_entry = PredictionEntry()
if ACTIVATE_NS:
    prediction_entry.enabled = False
    CorrectionEntry(prediction_entry)
end
continue
