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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_constant_velocity_request_point.json"
))
TARGET_NS = int(os.environ["NAVVIS_PROBE_TARGET_NS"])
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


class PredictionRequest(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x4C97C0), internal=True)
        self.condition = (
            "(*(unsigned long long*)$rsi == %d) && "
            "(*(unsigned long long*)$rdx == %d)"
            % (CONSTANT_VELOCITY_VPTR, TARGET_NS)
        )

    def stop(self):
        inferior = gdb.selected_inferior()
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            timestamp_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(timestamp_pointer, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if timestamp_ns != TARGET_NS:
            return False

        prediction_frame = gdb.newest_frame()
        transform_frame = prediction_frame.older()
        ray_loop_frame = (
            transform_frame.older() if transform_frame is not None else None
        )
        record = {"timestamp_ns": timestamp_ns}
        if ray_loop_frame is not None:
            point_address = int(ray_loop_frame.read_register("rbx"))
            neighborhood_address = point_address - 32
            neighborhood = bytes(
                inferior.read_memory(neighborhood_address, 96)
            )
            record.update({
                "point_address": point_address,
                "point_raw_hex": bytes(
                    inferior.read_memory(point_address, 32)
                ).hex(),
                "point_float32x6": struct.unpack(
                    "<6f", bytes(inferior.read_memory(point_address, 24))
                ),
                "point_timestamp_ns": struct.unpack(
                    "<Q", bytes(inferior.read_memory(point_address + 24, 8))
                )[0],
                "neighborhood_start_address": neighborhood_address,
                "neighborhood_raw_hex": neighborhood.hex(),
                "ray_loop_pc_offset": int(ray_loop_frame.pc()) - BASE,
            })
        OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
        gdb.write("captured request point at %d ns\n" % timestamp_ns)
        gdb.execute("quit")
        return False


class PredictorCorrectionEntry(gdb.Breakpoint):
    def __init__(self, request_breakpoint):
        super().__init__("*%#x" % (BASE + 0x4CA2D0), internal=True)
        self.request_breakpoint = request_breakpoint

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
        self.request_breakpoint.enabled = True
        self.enabled = False
        return False


request_breakpoint = PredictionRequest()
if ACTIVATE_NS:
    request_breakpoint.enabled = False
    PredictorCorrectionEntry(request_breakpoint)
end
continue
