set pagination off
set confirm off
starti
python
import gdb
import json
import struct

# Resolved from the PointCloudMinRangeFilter vtable in the installed G11
# surveyorslam_processing_node (vptr image offset 0x1825798).
IMAGE_BASE = 0x555555554000
PROCESS = IMAGE_BASE + 0x3bc5f0
INDEX = "/tmp/navvis_vendor_minimum_range_filter.jsonl"
MAX_HITS = 30
open(INDEX, "w").close()

def read_u64(address):
    return struct.unpack(
        "<Q", bytes(gdb.selected_inferior().read_memory(address, 8))
    )[0]

def save_cloud(path, cloud):
    inferior = gdb.selected_inferior()
    begin = read_u64(cloud + 0x30)
    end = read_u64(cloud + 0x38)
    size = end - begin
    if size < 0 or size % 32 or size > 512 * 1024 * 1024:
        raise RuntimeError("unexpected PointXYZITR vector size %d" % size)
    header = bytes(inferior.read_memory(cloud, 0x90))
    points = bytes(inferior.read_memory(begin, size)) if size else b""
    with open(path, "wb") as stream:
        stream.write(b"NVFLCLD1")
        stream.write(struct.pack("<QQ", cloud, size // 32))
        stream.write(header)
        stream.write(points)
    return size // 32

def append(record):
    with open(INDEX, "a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")

class Finish(gdb.FinishBreakpoint):
    def __init__(self, hit, owner, output_cloud, input_count):
        super().__init__(internal=True)
        self.hit = hit
        self.owner = owner
        self.output_cloud = output_cloud
        self.input_count = input_count

    def stop(self):
        path = "/tmp/navvis_vendor_minimum_range_%02d_out.bin" % self.hit
        output_count = save_cloud(path, self.output_cloud)
        append({"hit": self.hit, "owner": self.owner,
                "input_count": self.input_count,
                "output_count": output_count, "output": path})
        gdb.write("minimum-range stage %d owner=%#x %d -> %d\n" %
                  (self.hit, self.owner, self.input_count, output_count))
        return False

class Breakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % PROCESS, internal=True)
        self.hits = 0

    def stop(self):
        self.hits += 1
        if self.hits > MAX_HITS:
            return False
        owner = int(gdb.parse_and_eval("$rdi"))
        input_cloud = int(gdb.parse_and_eval("$rsi"))
        output_cloud = int(gdb.parse_and_eval("$rdx"))
        path = "/tmp/navvis_vendor_minimum_range_%02d_in.bin" % self.hits
        input_count = save_cloud(path, input_cloud)
        append({"hit": self.hits, "owner": owner,
                "owner_hex": bytes(gdb.selected_inferior().read_memory(owner, 0x98)).hex(),
                "input_count": input_count, "input": path})
        Finish(self.hits, owner, output_cloud, input_count)
        return False

Breakpoint()
end
continue
