set pagination off
set confirm off
starti
python
import gdb
import json
from pathlib import Path
import struct

OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_local_icp_call0")
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


def dump_raw(name, address, size):
    try:
        data = memory(address, size)
    except gdb.MemoryError:
        return False
    (OUTPUT / name).write_bytes(data)
    return True


def discover_vectors(label, address, size, metadata):
    try:
        data = memory(address, size)
    except gdb.MemoryError:
        return
    for offset in range(0, size - 23, 8):
        begin, end, capacity = struct.unpack_from("<QQQ", data, offset)
        used = end - begin
        allocated = capacity - begin
        if not (0x10000 <= begin <= end <= capacity and 0 < used <= allocated <= 16_000_000):
            continue
        # Point/surfel containers in this call have aligned element widths.
        possible_widths = [width for width in (4, 8, 12, 16, 24, 32, 48, 64, 104) if used % width == 0]
        if not possible_widths:
            continue
        name = "%s_%03x.bin" % (label, offset)
        if dump_raw(name, begin, used):
            metadata.append({
                "owner": label,
                "offset": offset,
                "begin": begin,
                "end": end,
                "capacity": capacity,
                "used_bytes": used,
                "possible_element_widths": possible_widths,
                "path": name,
            })


BASE = image_base("surveyorslam_processing_node")
OUTPUT.mkdir(parents=True, exist_ok=True)


class MatchReturn(gdb.FinishBreakpoint):
    def __init__(self, output_pointer, metadata):
        super().__init__(gdb.newest_frame(), internal=True)
        self.output_pointer = output_pointer
        self.metadata = metadata

    def stop(self):
        dump_raw("result_after.bin", self.output_pointer, 0x600)
        self.metadata["return_pc"] = int(gdb.parse_and_eval("$pc"))
        (OUTPUT / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True)
        )
        gdb.write("captured complete local ICP call\n")
        gdb.execute("quit")
        return False


class MatchBreakpoint(gdb.Breakpoint):
    def stop(self):
        registers = {
            name: int(gdb.parse_and_eval("$" + name))
            for name in ("rdi", "rsi", "rdx", "rcx", "r8", "r9", "rsp")
        }
        stack_argument = struct.unpack("<Q", memory(registers["rsp"] + 8, 8))[0]
        metadata = {
            "executable_base": BASE,
            "match_offset": MATCH_OFFSET,
            "registers": registers,
            "stack_argument": stack_argument,
            "vectors": [],
        }
        regions = (
            ("matcher", registers["rsi"], 0x300),
            ("source", registers["rdx"], 0x180),
            ("initial", registers["rcx"], 0x180),
            ("normalization", registers["r8"], 0x180),
            ("options", registers["r9"], 0x300),
            ("stack_argument", stack_argument, 0x300),
            ("result_before", registers["rdi"], 0x600),
        )
        for label, address, size in regions:
            dump_raw(label + ".bin", address, size)
            discover_vectors(label, address, size, metadata["vectors"])
        (OUTPUT / "metadata_entry.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
        MatchReturn(registers["rdi"], metadata)
        self.enabled = False
        return False


MatchBreakpoint("*%#x" % (BASE + MATCH_OFFSET), internal=True)
end
continue
