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


MATRIX_ENTRY_OFFSET = 0x3CBFE0
MATRIX_RETURN_OFFSET = 0x3CC113
OUTER_CALL_RETURN_OFFSET = 0x4A7B5C
PREDICTOR_CORRECTION_OFFSET = 0x4CA2D0
ACTIVATE_NS = int(os.environ.get(
    "NAVVIS_OUTER_MATRIX_ACTIVATE_NS", "1784626916407899431"
))
CAPTURE_LIMIT = int(os.environ.get("NAVVIS_OUTER_MATRIX_CAPTURE_LIMIT", "4"))
OUTPUT_DIR = Path(os.environ.get(
    "NAVVIS_OUTER_MATRIX_CAPTURE",
    "/tmp/navvis_vendor_outer_deskew_matrix",
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
pending = {}
records = []


class MatrixReturn(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        record = pending.pop(thread, None)
        if record is None:
            return False
        output_address, pose = record
        matrix = bytes(gdb.selected_inferior().read_memory(output_address, 64))
        index = len(records)
        (OUTPUT_DIR / ("pose_%02d.bin" % index)).write_bytes(pose)
        (OUTPUT_DIR / ("matrix_%02d.bin" % index)).write_bytes(matrix)
        records.append({
            "index": index,
            "pose": "pose_%02d.bin" % index,
            "matrix": "matrix_%02d.bin" % index,
        })
        (OUTPUT_DIR / "metadata.json").write_text(json.dumps(
            {
                "activate_ns": ACTIVATE_NS,
                "captures": records,
                "executable_base": base,
            },
            indent=2,
            sort_keys=True,
        ))
        if len(records) >= CAPTURE_LIMIT:
            gdb.write(
                "captured %d outer deskew matrices to %s\n"
                % (len(records), OUTPUT_DIR)
            )
            gdb.execute("quit")
        return False


class MatrixEntry(gdb.Breakpoint):
    def stop(self):
        stack_pointer = int(gdb.parse_and_eval("$rsp"))
        return_address = struct.unpack(
            "<Q",
            bytes(gdb.selected_inferior().read_memory(stack_pointer, 8)),
        )[0]
        if return_address != base + OUTER_CALL_RETURN_OFFSET:
            return False
        output_address = int(gdb.parse_and_eval("$rdi"))
        pose_address = int(gdb.parse_and_eval("$rsi"))
        pose = bytes(gdb.selected_inferior().read_memory(pose_address, 32))
        pending[gdb.selected_thread().num] = (output_address, pose)
        return False


matrix_entry = MatrixEntry(
    "*%#x" % (base + MATRIX_ENTRY_OFFSET), internal=True
)
matrix_entry.enabled = False
matrix_return = MatrixReturn(
    "*%#x" % (base + MATRIX_RETURN_OFFSET), internal=True
)
matrix_return.enabled = False


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
        matrix_entry.enabled = True
        matrix_return.enabled = True
        self.enabled = False
        gdb.write(
            "enabled outer deskew matrix capture after correction at %d ns\n"
            % timestamp_ns
        )
        return False


PredictorCorrectionEntry(
    "*%#x" % (base + PREDICTOR_CORRECTION_OFFSET), internal=True
)
end
continue
