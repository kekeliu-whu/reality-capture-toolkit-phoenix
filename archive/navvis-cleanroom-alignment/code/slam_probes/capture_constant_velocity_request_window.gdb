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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_constant_velocity_requests.json"
))
START_NS = int(os.environ["NAVVIS_PROBE_START_NS"])
END_NS = int(os.environ["NAVVIS_PROBE_END_NS"])
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


def flush():
    OUTPUT.write_text(json.dumps(records, indent=2) + "\n")


class PredictionRequest(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x4C97C0), internal=True)
        self.condition = (
            "(*(unsigned long long*)$rsi == %d) && "
            "(*(unsigned long long*)$rdx >= %d)"
            % (CONSTANT_VELOCITY_VPTR, START_NS)
        )

    def stop(self):
        inferior = gdb.selected_inferior()
        predictor = int(gdb.parse_and_eval("$rsi"))
        timestamp_pointer = int(gdb.parse_and_eval("$rdx"))
        try:
            timestamp_ns = struct.unpack(
                "<Q", bytes(inferior.read_memory(timestamp_pointer, 8))
            )[0]
            vptr = struct.unpack(
                "<Q", bytes(inferior.read_memory(predictor, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if vptr != CONSTANT_VELOCITY_VPTR or timestamp_ns < START_NS:
            return False
        if timestamp_ns > END_NS:
            flush()
            gdb.write(
                "captured %d constant-velocity requests in window\n"
                % len(records)
            )
            gdb.execute("quit")
            return False
        frame = gdb.newest_frame()
        callers = []
        for _ in range(6):
            frame = frame.older() if frame is not None else None
            if frame is None:
                break
            callers.append({
                "pc_offset": int(frame.pc()) - BASE,
                "name": frame.name(),
            })
        records.append({
            "timestamp_ns": timestamp_ns,
            "predictor": predictor,
            "callers": callers,
        })
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
        gdb.write(
            "enabled constant-velocity request-window probe at %d ns\n"
            % timestamp_ns
        )
        return False


request_breakpoint = PredictionRequest()
if ACTIVATE_NS:
    request_breakpoint.enabled = False
    PredictorCorrectionEntry(request_breakpoint)
end
continue
