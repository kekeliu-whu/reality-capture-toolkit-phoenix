set pagination off
set confirm off
starti
python
import gdb
import json
import struct

IMAGE_BASE = 0x555555554000
PROCESS = IMAGE_BASE + 0x3b9b20
INDEX = "/tmp/navvis_vendor_filter_drops.jsonl"
MAX_HITS = 120
open(INDEX, "w").close()

def read_u64(address):
    return struct.unpack("<Q", bytes(gdb.selected_inferior().read_memory(address, 8)))[0]

def read_cloud(cloud):
    inferior = gdb.selected_inferior()
    begin = read_u64(cloud + 0x30)
    end = read_u64(cloud + 0x38)
    size = end - begin
    if size < 0 or size % 32 or size > 512 * 1024 * 1024:
        raise RuntimeError("unexpected PointXYZITR vector size %d" % size)
    header = bytes(inferior.read_memory(cloud, 0x90))
    points = bytes(inferior.read_memory(begin, size)) if size else b""
    return header, points

def save_cloud(path, cloud, header, points):
    with open(path, "wb") as stream:
        stream.write(b"NVFLCLD1")
        stream.write(struct.pack("<QQ", cloud, len(points) // 32))
        stream.write(header)
        stream.write(points)

def append(record):
    with open(INDEX, "a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")

class DropFinish(gdb.FinishBreakpoint):
    def __init__(self, hit, owner, input_cloud, output_cloud, header, points):
        super().__init__(internal=True)
        self.hit = hit
        self.owner = owner
        self.input_cloud = input_cloud
        self.output_cloud = output_cloud
        self.header = header
        self.points = points

    def stop(self):
        output_header, output_points = read_cloud(self.output_cloud)
        input_count = len(self.points) // 32
        output_count = len(output_points) // 32
        if output_count != input_count:
            input_path = "/tmp/navvis_vendor_filter_drop_%03d_in.bin" % self.hit
            output_path = "/tmp/navvis_vendor_filter_drop_%03d_out.bin" % self.hit
            save_cloud(input_path, self.input_cloud, self.header, self.points)
            save_cloud(output_path, self.output_cloud, output_header, output_points)
            append({"hit": self.hit, "owner": self.owner,
                    "input_count": input_count, "output_count": output_count,
                    "input": input_path, "output": output_path})
            gdb.write("filter drop %d owner=%#x %d -> %d\n" %
                      (self.hit, self.owner, input_count, output_count))
        return False

class DropBreakpoint(gdb.Breakpoint):
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
        header, points = read_cloud(input_cloud)
        DropFinish(self.hits, owner, input_cloud, output_cloud, header, points)
        return False

DropBreakpoint()
end
continue
