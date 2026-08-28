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


OUTER_MATRIX_RETURN_OFFSET = 0x4A7B5C
PREDICTOR_CORRECTION_OFFSET = 0x4CA2D0
ACTIVATE_NS = int(os.environ.get(
    "NAVVIS_OUTER_BATCH_ACTIVATE_NS", "1784626916407899431"
))
OUTPUT_DIR = Path(os.environ.get(
    "NAVVIS_OUTER_BATCH_CAPTURE",
    "/tmp/navvis_vendor_outer_deskew_batch",
))


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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class OuterMatrixReturn(gdb.Breakpoint):
    def stop(self):
        stack_pointer = int(gdb.parse_and_eval("$rsp"))
        begin, end = struct.unpack(
            "<QQ",
            bytes(gdb.selected_inferior().read_memory(stack_pointer + 0x40, 16)),
        )
        byte_count = end - begin
        if byte_count <= 0 or byte_count % 24 != 0:
            raise RuntimeError(
                "unexpected range-measurement vector: %#x--%#x" % (begin, end)
            )
        measurements = bytes(
            gdb.selected_inferior().read_memory(begin, byte_count)
        )
        matrix = bytes(
            gdb.selected_inferior().read_memory(stack_pointer + 0x1E0, 64)
        )
        pose = bytes(
            gdb.selected_inferior().read_memory(stack_pointer + 0x100, 32)
        )
        (OUTPUT_DIR / "measurements.bin").write_bytes(measurements)
        (OUTPUT_DIR / "matrix.bin").write_bytes(matrix)
        (OUTPUT_DIR / "pose.bin").write_bytes(pose)
        (OUTPUT_DIR / "metadata.json").write_text(json.dumps(
            {
                "activate_ns": ACTIVATE_NS,
                "byte_count": byte_count,
                "measurement_count": byte_count // 24,
                "executable_base": base,
            },
            indent=2,
            sort_keys=True,
        ))
        gdb.write(
            "captured %d outer deskew measurements to %s\n"
            % (byte_count // 24, OUTPUT_DIR)
        )
        gdb.execute("quit")
        return False


outer_return = OuterMatrixReturn(
    "*%#x" % (base + OUTER_MATRIX_RETURN_OFFSET), internal=True
)
outer_return.enabled = False


class PredictorCorrectionEntry(gdb.Breakpoint):
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
        outer_return.enabled = True
        self.enabled = False
        gdb.write(
            "enabled outer deskew batch capture after correction at %d ns\n"
            % timestamp_ns
        )
        return False


PredictorCorrectionEntry(
    "*%#x" % (base + PREDICTOR_CORRECTION_OFFSET), internal=True
)
end
continue
