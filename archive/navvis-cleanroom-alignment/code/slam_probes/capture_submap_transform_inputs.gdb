set pagination off
set confirm off
set breakpoint pending on
starti
python
import gdb
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ.get("NAVVIS_PROBE_OUTPUT", "/tmp/navvis_submap_transform"))
CALL_LIMIT = int(os.environ.get("NAVVIS_PROBE_CALL_LIMIT", "20"))
TRANSFORM_OFFSET = 0x646940


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


def memory(address, size):
    return bytes(gdb.selected_inferior().read_memory(address, size))


OUTPUT.mkdir(parents=True, exist_ok=True)
records = []


class TransformInput(gdb.Breakpoint):
    def stop(self):
        index = len(records)
        vector = int(gdb.parse_and_eval("$rsi"))
        pose = int(gdb.parse_and_eval("$rdx"))
        begin, end, capacity = struct.unpack("<QQQ", memory(vector, 24))
        if not begin <= end <= capacity or (end - begin) % 24 != 0:
            raise RuntimeError("invalid RangeMeasurement vector")
        ray_path = "batch_%02d_input.bin" % index
        pose_path = "batch_%02d_float_pose.bin" % index
        (OUTPUT / ray_path).write_bytes(memory(begin, end - begin))
        (OUTPUT / pose_path).write_bytes(memory(pose, 32))
        records.append({
            "index": index,
            "count": (end - begin) // 24,
            "begin": begin,
            "end": end,
            "ray_path": ray_path,
            "pose_path": pose_path,
        })
        (OUTPUT / "metadata.json").write_text(
            json.dumps({"calls": records}, indent=2, sort_keys=True) + "\n"
        )
        if len(records) >= CALL_LIMIT:
            gdb.write("captured %d submap transform inputs\n" % len(records))
            gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
TransformInput("*%#x" % (base + TRANSFORM_OFFSET), internal=True)
end
continue
