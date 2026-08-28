set pagination off
set confirm off
set print elements 0

# Start at the dynamic loader so that the PIE mapping exists, then resolve the
# renderer base from /proc instead of relying on ASLR being disabled.
starti

python
import gdb
import os
import struct

inferior = gdb.selected_inferior()
exe = os.path.realpath(gdb.current_progspace().filename)
base = None
with open(f"/proc/{inferior.pid}/maps", "r", encoding="ascii") as maps_file:
    for line in maps_file:
        fields = line.split()
        if len(fields) < 6 or os.path.realpath(fields[-1]) != exe:
            continue
        mapping_start = int(fields[0].split("-", 1)[0], 16)
        file_offset = int(fields[2], 16)
        candidate = mapping_start - file_offset
        base = candidate if base is None else min(base, candidate)

if base is None:
    raise gdb.GdbError(f"could not locate PIE base for {exe}")

gdb.write(f"renderer PIE base: 0x{base:x}\n")

class ExposureReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, output_address):
        super().__init__(frame, internal=True)
        self.output_address = output_address

    def stop(self):
        raw = bytes(inferior.read_memory(self.output_address, 24))
        begin, end, capacity = struct.unpack("<QQQ", raw)
        gdb.write(
            "soft-exposure vector: "
            f"object=0x{self.output_address:x}, begin=0x{begin:x}, "
            f"end=0x{end:x}, capacity=0x{capacity:x}\n"
        )
        if begin == 0 or end < begin or end - begin > 4096:
            gdb.write("soft-exposure vector layout was not recognized\n")
            return True
        payload = bytes(inferior.read_memory(begin, end - begin))
        values = struct.unpack("<" + "f" * (len(payload) // 4), payload)
        gdb.write("soft-exposure raw floats: " + repr(values) + "\n")
        return True

class ExposureEntry(gdb.Breakpoint):
    def stop(self):
        # System V C++ ABI: std::vector<Eigen::Vector3f> is returned through a
        # hidden storage pointer in RDI. RSI is the object instance.
        output_address = int(gdb.parse_and_eval("$rdi"))
        gdb.write(
            "entered ExposureCompensatorSoftConstraint::computeGainValuesImpl "
            f"at {self.location}; output=0x{output_address:x}\n"
        )
        ExposureReturn(gdb.newest_frame(), output_address)
        return False

# Reverse-engineered virtual function offset in build 05ad952633e1dadaf52fb40c2187932e20f8a5ee.
ExposureEntry(f"*0x{base + 0x191370:x}")
end

continue
