set pagination off
set confirm off
set print thread-events off
starti
python
import gdb
import os
import struct


TRANSFORM_OFFSET = 0x4B85B0
FIRST_RETURN_OFFSET = 0x4A7767
SECOND_RETURN_OFFSET = 0x4A7777
ADVANCE_OFFSET = 0x6E0F30
RAW_VPTR_OFFSET = 0x1830660
START_RAY_TIME_NS = int(os.environ.get(
    "NAVVIS_DESKEW_START_TIME_NS", "1784626878164344000"
))
ACTIVATE_NS = int(os.environ.get("NAVVIS_DESKEW_ACTIVATE_NS", "0"))
CAPTURE_LIMIT = int(os.environ.get("NAVVIS_DESKEW_CAPTURE_LIMIT", "2048"))
OUTPUT_PATH = os.environ.get(
    "NAVVIS_DESKEW_CAPTURE",
    "/tmp/navvis_vendor_deskew_float_calls.bin",
)


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if (
                len(fields) >= 6
                and fields[2] == "00000000"
                and fragment in fields[-1]
            ):
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


base = image_base("surveyorslam_processing_node")
output = open(OUTPUT_PATH, "wb")
captured = 0
pending = {}


class TransformReturn(gdb.Breakpoint):
    def stop(self):
        global captured
        thread = gdb.selected_thread().num
        record = pending.pop(thread, None)
        if record is None:
            return False
        output_address, prefix = record
        result = bytes(
            gdb.selected_inferior().read_memory(output_address, 12)
        )
        output.write(prefix + result)
        captured += 1
        if captured >= CAPTURE_LIMIT:
            output.flush()
            output.close()
            gdb.write(
                "captured %d vendor deskew transforms to %s\n"
                % (captured, OUTPUT_PATH)
            )
            gdb.execute("quit")
        return False


class TransformEntry(gdb.Breakpoint):
    def stop(self):
        return_address = struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(int(gdb.parse_and_eval("$rsp")), 8))
        )[0]
        if return_address == base + FIRST_RETURN_OFFSET:
            call_site = 0
        elif return_address == base + SECOND_RETURN_OFFSET:
            call_site = 1
        else:
            return False

        output_address = int(gdb.parse_and_eval("$rdi"))
        pose_address = int(gdb.parse_and_eval("$rsi"))
        input_address = int(gdb.parse_and_eval("$rdx"))
        point = bytes(gdb.selected_inferior().read_memory(input_address, 12))
        pose = bytes(gdb.selected_inferior().read_memory(pose_address, 32))
        prefix = struct.pack("<B3x", call_site) + point + pose
        pending[gdb.selected_thread().num] = (output_address, prefix)
        return False


transform_breakpoint = TransformEntry(
    "*%#x" % (base + TRANSFORM_OFFSET), internal=True
)
transform_breakpoint.enabled = False
transform_return_breakpoint = TransformReturn(
    "*%#x" % (base + TRANSFORM_OFFSET + 0xD3), internal=True
)
transform_return_breakpoint.enabled = False


class FirstRetainedRay(gdb.Breakpoint):
    def stop(self):
        owner = int(gdb.parse_and_eval("$rdi"))
        timestamp_ns = int(gdb.parse_and_eval("$rsi")) & 0xFFFFFFFFFFFFFFFF
        vptr = struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(owner, 8))
        )[0]
        if (
            vptr != base + RAW_VPTR_OFFSET
            or timestamp_ns < START_RAY_TIME_NS
        ):
            return False
        transform_breakpoint.enabled = True
        transform_return_breakpoint.enabled = True
        self.delete()
        gdb.write(
            "enabled deskew capture at raw tracker time %d ns\n"
            % timestamp_ns
        )
        return False


first_retained_ray = FirstRetainedRay(
    "*%#x" % (base + ADVANCE_OFFSET), internal=True
)


class PredictorCorrectionEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (base + 0x4CA2D0), internal=True)

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
        transform_breakpoint.enabled = True
        transform_return_breakpoint.enabled = True
        self.enabled = False
        gdb.write(
            "enabled deskew transform capture after correction at %d ns\n"
            % timestamp_ns
        )
        return False


if ACTIVATE_NS:
    first_retained_ray.enabled = False
    PredictorCorrectionEntry()
end
continue
