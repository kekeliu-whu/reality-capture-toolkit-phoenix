set pagination off
set confirm off
starti
python
import gdb
import json
from pathlib import Path
import struct


OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_validity_region.json")
VALIDITY_OFFSET = 0x4985D0
MATCH_OFFSET = 0x6C4660


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


records = []


class ValidityEntry(gdb.Breakpoint):
    def stop(self):
        surfel = int(gdb.parse_and_eval("$rsi"))
        raw = memory(surfel, 104)
        center = struct.unpack_from("<3f", raw, 0x10)
        if not (
            -0.65 <= center[0] <= 1.05
            and -2.85 <= center[1] <= -0.75
            and 0.65 <= center[2] <= 1.15
        ):
            return False
        thresholds = int(gdb.parse_and_eval("$rdi"))
        record = {
            "surfel": surfel,
            "caller": int(gdb.newest_frame().older().pc()),
            "voxel_scale": float(gdb.parse_and_eval("$xmm0.v4_float[0]")),
            "thresholds": list(struct.unpack("<6f", memory(thresholds, 24))),
            "weight": struct.unpack_from("<f", raw, 0x08)[0],
            "count": struct.unpack_from("<I", raw, 0x0c)[0],
            "center": list(center),
            "covariance_memory": list(struct.unpack_from("<9f", raw, 0x1c)),
            "viewpoint_mean": list(struct.unpack_from("<3f", raw, 0x40)),
            "normal": list(struct.unpack_from("<3f", raw, 0x4c)),
            "eigenvalues": list(struct.unpack_from("<3f", raw, 0x58)),
            "dirty": bool(raw[0x64]),
        }
        records.append(record)
        return False


class FirstMatch(gdb.Breakpoint):
    def stop(self):
        OUTPUT.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        gdb.write("captured %d regional surfel-validity calls\n" % len(records))
        gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
ValidityEntry("*%#x" % (base + VALIDITY_OFFSET), internal=True)
FirstMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
