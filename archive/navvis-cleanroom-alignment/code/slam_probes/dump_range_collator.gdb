set pagination off
set confirm off
starti
python
import gdb
import struct

OUTPUT = "/tmp/navvis_vendor_range_inputs.bin"
IMAGE_BASE = 0x555555554000

with open(OUTPUT, "wb") as stream:
    stream.write(b"NVRANGE1")

class RangeInputBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x6426e0), internal=True)
        self.hits = 0

    def stop(self):
        data = int(gdb.parse_and_eval("$rdx"))
        sensor = int(gdb.parse_and_eval("$rsi"))
        inferior = gdb.selected_inferior()
        header = bytes(inferior.read_memory(data, 24))
        timestamp, begin, end = struct.unpack("<qQQ", header)
        count = (end - begin) // 28
        rays = bytes(inferior.read_memory(begin, count * 28))
        with open(OUTPUT, "ab") as stream:
            stream.write(struct.pack("<qQII", timestamp, sensor, count, len(rays)))
            stream.write(rays)
        gdb.write(
            "DUMP hit=%d time_ticks=%d rays=%d sensor_arg=%#x\n"
            % (self.hits, timestamp, count, sensor)
        )
        self.hits += 1
        if self.hits >= 7:
            gdb.execute("quit")
        return False

RangeInputBreakpoint()
end
continue
