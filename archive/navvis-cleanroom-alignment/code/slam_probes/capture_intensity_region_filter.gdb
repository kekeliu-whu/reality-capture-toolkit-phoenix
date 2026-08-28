set pagination off
set confirm off
starti
python
import gdb
import json
import struct

OUTPUT = "/tmp/navvis_vendor_intensity_region_filter.jsonl"
IMAGE_BASE = 0x555555554000
CONSTRUCTOR = IMAGE_BASE + 0x3a4980
PROCESS = IMAGE_BASE + 0x3b9b20

open(OUTPUT, "w").close()

def append_record(record):
    with open(OUTPUT, "a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")

def read_hex(address, size):
    if not address:
        return None
    try:
        return bytes(gdb.selected_inferior().read_memory(address, size)).hex()
    except gdb.MemoryError:
        return None

def u64_at(address):
    raw = read_hex(address, 8)
    if raw is None:
        return 0
    return struct.unpack("<Q", bytes.fromhex(raw))[0]

class ConstructorFinish(gdb.FinishBreakpoint):
    def __init__(self, sequence, destination, args):
        super().__init__(internal=True)
        self.sequence = sequence
        self.destination = destination
        self.args = args

    def stop(self):
        raw_pointer = u64_at(self.destination)
        control_pointer = u64_at(self.destination + 8)
        append_record({
            "kind": "constructor_return",
            "sequence": self.sequence,
            "destination": self.destination,
            "raw_pointer": raw_pointer,
            "control_pointer": control_pointer,
            "args": self.args,
            "object_hex": read_hex(raw_pointer, 0xa0),
            "control_hex": read_hex(control_pointer, 0x40),
        })
        return False

class ConstructorBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % CONSTRUCTOR, internal=True)
        self.hits = 0

    def stop(self):
        self.hits += 1
        destination = int(gdb.parse_and_eval("$rdi"))
        region_argument = int(gdb.parse_and_eval("$rdx"))
        args = {
            "flag_a": int(gdb.parse_and_eval("$esi")) & 0xff,
            "flag_b": int(gdb.parse_and_eval("$ecx")) & 0xff,
            "radius": float(gdb.parse_and_eval("$xmm0.v4_float[0]")),
            "region_argument": region_argument,
            "region_argument_hex": read_hex(region_argument, 0x80),
        }
        append_record({
            "kind": "constructor_entry",
            "sequence": self.hits,
            "destination": destination,
            "args": args,
        })
        ConstructorFinish(self.hits, destination, args)
        return False

class ProcessBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % PROCESS, internal=True)
        self.hits = 0
        self.seen = set()

    def stop(self):
        self.hits += 1
        owner = int(gdb.parse_and_eval("$rdi"))
        input_cloud = int(gdb.parse_and_eval("$rsi"))
        if owner not in self.seen or self.hits <= 12:
            self.seen.add(owner)
            pointers = {}
            for offset in (0x20, 0x28, 0x30, 0x38, 0x40, 0x48,
                           0x58, 0x60, 0x68, 0x70, 0x78, 0x80,
                           0x90, 0x98):
                value = u64_at(owner + offset)
                pointers["%02x" % offset] = {
                    "value": value,
                    "hex": read_hex(value, 0x80),
                }
            append_record({
                "kind": "process",
                "hit": self.hits,
                "owner": owner,
                "object_hex": read_hex(owner, 0xa0),
                "input_cloud": input_cloud,
                "input_cloud_hex": read_hex(input_cloud, 0xa0),
                "pointers": pointers,
            })
        if self.hits % 50 == 0:
            gdb.write("IntensityRegionFilter process hits=%d objects=%d\n" %
                      (self.hits, len(self.seen)))
        return False

ConstructorBreakpoint()
ProcessBreakpoint()
end
continue
