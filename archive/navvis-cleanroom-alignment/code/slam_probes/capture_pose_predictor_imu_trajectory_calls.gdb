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
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_pose_predictor_imu_calls.json"
))
TARGET_NS = int(os.environ.get("NAVVIS_PROBE_TARGET_NS", "0"))
ANY_VPTR = os.environ.get("NAVVIS_PROBE_ANY_VPTR", "0") != "0"


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


BASE = image_base("surveyorslam_processing_node")
PREDICTOR_VPTR = BASE + 0x1827C40
METHOD_OFFSETS = (
    0x4CB550,
    0x4CB720,
    0x4C9500,
    0x4C90F0,
    0x4C91A0,
    0x4CA130,
    0x4C97C0,
    0x4CA2D0,
    0x4CA970,
    0x4B0490,
    0x4C9100,
    0x4C9110,
    0x4C9170,
    0x4C9200,
    0x4C8D30,
    0x4C8D20,
    0x4C8CF0,
    0x4C8CC0,
    0x4C8DB0,
    0x4C8F00,
    0x4C8C60,
    0x4C8C50,
    0x4C8BF0,
    0x4C8B90,
    0x4C8B30,
    0x4CBF40,
    0x4CB900,
    0x4CBD50,
    0x4CC0D0,
    0x4CC990,
    0x4CB9B0,
    0x4CB910,
    0x4CB920,
    0x4CB980,
    0x4CBA10,
    0x4C8460,
)
records = []


def flush():
    OUTPUT.write_text(json.dumps(records, indent=2) + "\n")


class MethodBreakpoint(gdb.Breakpoint):
    def __init__(self, offset):
        super().__init__("*%#x" % (BASE + offset), internal=True)
        self.offset = offset

    def stop(self):
        inferior = gdb.selected_inferior()
        owner = int(gdb.parse_and_eval("$rdi"))
        try:
            vptr = struct.unpack(
                "<Q", bytes(inferior.read_memory(owner, 8))
            )[0]
        except gdb.MemoryError:
            return False
        if not ANY_VPTR and vptr != PREDICTOR_VPTR:
            return False
        registers = {
            name: int(gdb.parse_and_eval("$" + name))
            for name in ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
        }
        record = {
            "method_offset": self.offset,
            "registers": registers,
        }
        records.append(record)
        if len(records) <= 32 or len(records) % 128 == 0:
            flush()
        timestamp = registers["rsi"]
        if TARGET_NS and TARGET_NS <= timestamp < (1 << 63):
            flush()
            gdb.write(
                "captured IMU trajectory predictor method %#x at %d ns\n"
                % (self.offset, timestamp)
            )
            gdb.execute("quit")
        return False


for offset in METHOD_OFFSETS:
    MethodBreakpoint(offset)
end
continue
