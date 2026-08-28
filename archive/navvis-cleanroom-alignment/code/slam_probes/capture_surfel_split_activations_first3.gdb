set pagination off
set confirm off
starti
python
import gdb
import json
from pathlib import Path
import struct


OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_split_activations_first3.json")
ACTIVATION_OFFSETS = {
    0x48FE9E: "r12",
    0x48FF01: "r12",
    0x490071: "r12",
    0x4992C2: "rbp",
    0x499311: "rbp",
    0x499451: "rbp",
}
MATCH_OFFSET = 0x6C4660
MATCH_LIMIT = 3


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


def surfel_fields(voxel):
    raw = memory(voxel, 232)
    return {
        "voxel": voxel,
        "primary_weight": struct.unpack_from("<f", raw, 0x10)[0],
        "primary_count": struct.unpack_from("<I", raw, 0x14)[0],
        "primary_center": list(struct.unpack_from("<3f", raw, 0x18)),
        "primary_normal": list(struct.unpack_from("<3f", raw, 0x54)),
        "primary_eigenvalues": list(struct.unpack_from("<3f", raw, 0x60)),
        "secondary_weight": struct.unpack_from("<f", raw, 0x78)[0],
        "secondary_count": struct.unpack_from("<I", raw, 0x7c)[0],
        "secondary_center": list(struct.unpack_from("<3f", raw, 0x80)),
    }


records = []
match_count = 0


class Activation(gdb.Breakpoint):
    def __init__(self, offset, register):
        super().__init__("*%#x" % (base + offset), internal=True)
        self.offset = offset
        self.register = register

    def stop(self):
        voxel = int(gdb.parse_and_eval("$" + self.register))
        record = surfel_fields(voxel)
        record.update({
            "before_match": match_count,
            "instruction_offset": self.offset,
            "thread": gdb.selected_thread().num,
        })
        records.append(record)
        return False


class LocalMatch(gdb.Breakpoint):
    def stop(self):
        global match_count
        match_count += 1
        OUTPUT.write_text(json.dumps({
            "match_count": match_count,
            "activations": records,
        }, indent=2, sort_keys=True) + "\n")
        if match_count >= MATCH_LIMIT:
            gdb.write("captured %d surfel split activations\n" % len(records))
            gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
for offset, register in ACTIVATION_OFFSETS.items():
    Activation(offset, register)
LocalMatch("*%#x" % (base + MATCH_OFFSET), internal=True)
end
continue
