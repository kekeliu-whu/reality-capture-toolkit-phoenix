set pagination off
set confirm off
starti
python
import gdb
import json

OUTPUT = "/tmp/navvis_scan_matcher_entries.jsonl"
OFFSETS = (
    0x6C1FA0, 0x6C2250, 0x6C2320, 0x6C27A0, 0x6C2820, 0x6C2A30,
    0x6C2C00, 0x6C36F0, 0x6C39F0, 0x6C4480, 0x6C4660, 0x6C5A90,
    0x6C5DA0, 0x6C7280, 0x6C7480, 0x6C7840, 0x6C7B20, 0x6C7FF0,
    0x6C81A0, 0x6C8640, 0x6C92E0, 0x6C9740, 0x6C9D70, 0x6CB2C0,
    0x6CB6C0, 0x6CB810, 0x6CC2B0, 0x6CC560, 0x6CC890, 0x6CCD00,
    0x6CD630, 0x6CDC40, 0x6CE800, 0x6CF030,
)


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


BASE = image_base("surveyorslam_processing_node")
open(OUTPUT, "w").close()
seen = set()


class EntryBreakpoint(gdb.Breakpoint):
    def __init__(self, offset):
        super().__init__("*%#x" % (BASE + offset), internal=True)
        self.offset = offset

    def stop(self):
        if self.offset in seen:
            return False
        seen.add(self.offset)
        caller = gdb.newest_frame().older()
        caller_pc = int(caller.pc()) if caller is not None else 0
        record = {
            "offset": self.offset,
            "caller_pc": caller_pc,
            "caller_offset": caller_pc - BASE if caller_pc >= BASE else None,
            "caller_name": None if caller is None else caller.name(),
            "rdi": int(gdb.parse_and_eval("$rdi")),
            "rsi": int(gdb.parse_and_eval("$rsi")),
            "rdx": int(gdb.parse_and_eval("$rdx")),
            "rcx": int(gdb.parse_and_eval("$rcx")),
            "r8": int(gdb.parse_and_eval("$r8")),
            "r9": int(gdb.parse_and_eval("$r9")),
        }
        with open(OUTPUT, "a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return False


for offset in OFFSETS:
    EntryBreakpoint(offset)
gdb.write("installed scan-matcher entry probes at executable base %#x\n" % BASE)
end
continue
