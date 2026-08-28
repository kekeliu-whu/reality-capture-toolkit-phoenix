set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
import os
import struct

inferior = gdb.selected_inferior()
executable = os.path.realpath(gdb.current_progspace().filename)
pie_base = None
with open(f"/proc/{inferior.pid}/maps", "r", encoding="ascii") as maps_file:
    for line in maps_file:
        fields = line.split()
        if len(fields) < 6 or os.path.realpath(fields[-1]) != executable:
            continue
        mapping_start = int(fields[0].split("-", 1)[0], 16)
        file_offset = int(fields[2], 16)
        candidate = mapping_start - file_offset
        pie_base = candidate if pie_base is None else min(pie_base, candidate)

if pie_base is None:
    raise gdb.GdbError("could not locate renderer PIE base")

target_index = 1448
state = {"input": 0, "output": 0, "count": 0}


def read_i32(address):
    return struct.unpack("<i", inferior.read_memory(address, 4))[0]


def read_u64(address):
    return struct.unpack("<Q", inferior.read_memory(address, 8))[0]


class HessianEntry(gdb.Breakpoint):
    def stop(self):
        system = int(gdb.selected_frame().read_register("rdi"))
        rows = read_i32(system + 8)
        columns = read_i32(system + 12)
        if rows * columns != 8192:
            return False
        input_object = int(gdb.selected_frame().read_register("rdx"))
        output_object = int(gdb.selected_frame().read_register("rcx"))
        state["input"] = read_u64(input_object)
        state["output"] = read_u64(output_object)
        state["count"] = rows * columns
        return False


class InteriorFirstLaneBeforeFinalAdd(gdb.Breakpoint):
    def stop(self):
        if state["count"] != 8192:
            return False
        frame = gdb.selected_frame()
        destination = (
            int(frame.read_register("r10"))
            + int(frame.read_register("rcx")) * 8
            - 8
        )
        if destination != state["output"] + target_index * 8:
            return False
        gdb.write(
            f"hessian-runtime target={target_index} destination=0x{destination:x}\n"
        )
        for index in (
            target_index,
            target_index - 1,
            target_index + 1,
            target_index - 128,
            target_index + 128,
        ):
            raw = read_u64(state["input"] + index * 8)
            value = struct.unpack("<d", struct.pack("<Q", raw))[0]
            gdb.write(f"input[{index}]={value:.17g} bits=0x{raw:016x}\n")
        gdb.execute("info registers rax rdi r10 rcx")
        gdb.execute(
            "info registers xmm7 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15"
        )
        return True


HessianEntry(f"*0x{pie_base + 0x20b140:x}")
InteriorFirstLaneBeforeFinalAdd(f"*0x{pie_base + 0x20b5be:x}")
end

continue
