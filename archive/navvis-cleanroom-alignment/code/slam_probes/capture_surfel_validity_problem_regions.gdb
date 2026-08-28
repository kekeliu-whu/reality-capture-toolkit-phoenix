set pagination off
set confirm off
starti
python
import gdb
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT",
    "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_validity_problem_regions.json",
))
VALIDITY_OFFSET = 0x4985D0
MATCH_OFFSET = 0x6C4660
MATCH_LIMIT = int(os.environ.get("NAVVIS_PROBE_MATCH_LIMIT", "3"))
MATCH_START = int(os.environ.get("NAVVIS_PROBE_MATCH_START", "0"))
REGION_RADIUS = float(os.environ.get("NAVVIS_PROBE_REGION_RADIUS", "0.55"))
centers_text = os.environ.get("NAVVIS_PROBE_CENTERS", "")
if centers_text:
    CENTERS = tuple(
        tuple(float(component) for component in center.split(","))
        for center in centers_text.split(";")
    )
else:
    CENTERS = (
        (-0.6030, -3.1042, 0.6477),
        (0.8847, 1.1010, -0.7100),
        (-2.5834, 1.4888, -0.6323),
        (1.2086, 3.6618, 2.1920),
    )


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


def near_problem(center):
    return any(
        abs(center[0] - wanted[0]) <= REGION_RADIUS
        and abs(center[1] - wanted[1]) <= REGION_RADIUS
        and abs(center[2] - wanted[2]) <= REGION_RADIUS
        for wanted in CENTERS
    )


def surfel_fields(surfel):
    raw = memory(surfel, 104)
    return {
        "weight": struct.unpack_from("<f", raw, 0x08)[0],
        "count": struct.unpack_from("<I", raw, 0x0c)[0],
        "center": list(struct.unpack_from("<3f", raw, 0x10)),
        "covariance_memory": list(struct.unpack_from("<9f", raw, 0x1c)),
        "viewpoint_mean": list(struct.unpack_from("<3f", raw, 0x40)),
        "normal": list(struct.unpack_from("<3f", raw, 0x4c)),
        "eigenvalues": list(struct.unpack_from("<3f", raw, 0x58)),
        "dirty": bool(raw[0x64]),
    }


records = []
match_count = 0


class ValidityEntry(gdb.Breakpoint):
    def __init__(self, specification):
        super().__init__(specification, internal=True)
        self.enabled = MATCH_START <= 0

    def stop(self):
        surfel = int(gdb.parse_and_eval("$rsi"))
        fields = surfel_fields(surfel)
        if match_count < MATCH_START or not near_problem(fields["center"]):
            return False
        thresholds = int(gdb.parse_and_eval("$rdi"))
        record = {
            "before_match": match_count,
            "surfel": surfel,
            "caller": int(gdb.newest_frame().older().pc()),
            "caller_offset": int(gdb.newest_frame().older().pc()) - base,
            "voxel_scale": float(gdb.parse_and_eval("$xmm0.v4_float[0]")),
            "thresholds": list(struct.unpack("<6f", memory(thresholds, 24))),
            "before": fields,
        }
        records.append(record)
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        if match_count >= MATCH_START:
            validity_breakpoint.enabled = True
        OUTPUT.write_text(json.dumps({
            "match_count": match_count,
            "records": records,
        }, indent=2, sort_keys=True) + "\n")
        if match_count >= MATCH_LIMIT:
            gdb.write("captured %d regional validity calls\n" % len(records))
            gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
validity_breakpoint = ValidityEntry("*%#x" % (base + VALIDITY_OFFSET))
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
