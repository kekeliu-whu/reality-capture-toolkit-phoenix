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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_imu_trajectory_prediction.json"
))
TARGET_NS = int(os.environ["NAVVIS_PROBE_TARGET_NS"])
ACTIVATE_NS = int(os.environ["NAVVIS_PROBE_ACTIVATE_NS"])


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


BASE = image_base("surveyorslam_processing_node")
IMU_TRAJECTORY_VPTR = BASE + 0x1827C40


class PredictionReturn(gdb.FinishBreakpoint):
    def __init__(self, output, predictor, timestamp_ns):
        super().__init__(internal=True)
        self.output = output
        self.predictor = predictor
        self.timestamp_ns = timestamp_ns

    def stop(self):
        inferior = gdb.selected_inferior()
        output_bytes = bytes(inferior.read_memory(self.output, 0x48))
        predictor_bytes = bytes(inferior.read_memory(self.predictor, 0x270))
        record = {
            "timestamp_ns": self.timestamp_ns,
            "prediction": {
                "translation": struct.unpack_from("<3d", output_bytes, 0),
                "quaternion_xyzw": struct.unpack_from("<4d", output_bytes, 0x20),
                "valid": output_bytes[0x40] != 0,
                "raw_hex": output_bytes.hex(),
            },
            "predictor": {
                "address": self.predictor,
                "vptr_offset": struct.unpack_from("<Q", predictor_bytes, 0)[0] - BASE,
                "current_time_ns": struct.unpack_from("<Q", predictor_bytes, 0x220)[0],
                "raw_hex": predictor_bytes.hex(),
            },
        }
        OUTPUT.write_text(json.dumps([record], indent=2) + "\n")
        gdb.write(
            "captured IMU-trajectory prediction at %d ns\n" % self.timestamp_ns
        )
        gdb.execute("quit")
        return False


class PredictionEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x4CC0D0), internal=True)

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
        if predictor_vptr != IMU_TRAJECTORY_VPTR or timestamp_ns != TARGET_NS:
            return False
        PredictionReturn(output, predictor, timestamp_ns)
        return False


class CorrectionEntry(gdb.Breakpoint):
    def __init__(self, prediction_entry):
        # Virtual slot 7 is the scan-matching correction path.  The ABI uses
        # rdi for its hidden return object, shifting `this` and the timestamp
        # pointer to rsi and rdx respectively.
        super().__init__("*%#x" % (BASE + 0x4CC990), internal=True)
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
        if predictor_vptr != IMU_TRAJECTORY_VPTR or timestamp_ns < ACTIVATE_NS:
            return False
        self.prediction_entry.enabled = True
        self.enabled = False
        gdb.write(
            "enabled IMU-trajectory point probe after correction at %d ns\n"
            % timestamp_ns
        )
        return False


prediction_entry = PredictionEntry()
prediction_entry.enabled = False
CorrectionEntry(prediction_entry)
end
continue
