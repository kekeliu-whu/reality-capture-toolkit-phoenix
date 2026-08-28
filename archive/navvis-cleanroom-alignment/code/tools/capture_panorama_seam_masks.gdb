set pagination off
set confirm off
set print elements 0
set print thread-events off

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


def read_u64(address):
    return struct.unpack("<Q", bytes(inferior.read_memory(address, 8)))[0]


def read_i32(address):
    return struct.unpack("<i", bytes(inferior.read_memory(address, 4)))[0]


class SeamPairEntry(gdb.Breakpoint):
    def stop(self):
        config = int(gdb.parse_and_eval("$rdi"))
        raw = bytes(inferior.read_memory(config, 0x48))
        terminal_cost, bad_region_penalty = struct.unpack_from("<ff", raw, 0x10)
        width, height = struct.unpack_from("<ii", raw, 0x18)
        shift = int(gdb.parse_and_eval("$r9"))
        shift_x, shift_y = struct.unpack(
            "<ff", bytes(inferior.read_memory(shift, 8)))
        gdb.write(
            "seam config: "
            f"terminal_cost={terminal_cost}, bad_region_penalty={bad_region_penalty}, "
            f"size={width}x{height}, shift=({shift_x}, {shift_y}), "
            f"raw={raw.hex()}\n"
        )
        self.enabled = False
        return False


class SeamMasksReady(gdb.Breakpoint):
    def stop(self):
        rsp = int(gdb.parse_and_eval("$rsp"))
        begin = read_u64(rsp + 0x30)
        end = read_u64(rsp + 0x38)
        count = (end - begin) // 96 if end >= begin else 0
        gdb.write(
            f"seam mask vector begin=0x{begin:x} end=0x{end:x} count={count}\n"
        )
        if count != 4:
            raise gdb.GdbError("unexpected seam mask vector layout")
        for index in range(count):
            mat = begin + index * 96
            rows = read_i32(mat + 8)
            cols = read_i32(mat + 12)
            data = read_u64(mat + 16)
            step_pointer = read_u64(mat + 72)
            row_step = read_u64(step_pointer)
            payload = bytearray(rows * cols)
            for row in range(rows):
                source = bytes(inferior.read_memory(data + row * row_step, cols))
                payload[row * cols:(row + 1) * cols] = source
            path = f"/tmp/navvis-original-seam-mask-cam{index}.pgm"
            with open(path, "wb") as output:
                output.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
                output.write(payload)
            gdb.write(
                f"wrote {path}: {cols}x{rows}, step={row_step}, "
                f"nonzero={sum(value != 0 for value in payload)}\n"
            )
        self.enabled = False
        return False


class SeamPairReturn(gdb.Breakpoint):
    stage = 0

    def stop(self):
        rsp = int(gdb.parse_and_eval("$rsp"))
        begin = read_u64(rsp + 0x30)
        end = read_u64(rsp + 0x38)
        count = (end - begin) // 96 if end >= begin else 0
        if count != 4:
            raise gdb.GdbError("unexpected intermediate seam mask vector layout")
        for index in range(count):
            mat = begin + index * 96
            rows = read_i32(mat + 8)
            cols = read_i32(mat + 12)
            data = read_u64(mat + 16)
            step_pointer = read_u64(mat + 72)
            row_step = read_u64(step_pointer)
            payload = bytearray(rows * cols)
            for row in range(rows):
                source = bytes(inferior.read_memory(data + row * row_step, cols))
                payload[row * cols:(row + 1) * cols] = source
            path = (
                f"/tmp/navvis-original-seam-stage{self.stage}-cam{index}.pgm"
            )
            with open(path, "wb") as output:
                output.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
                output.write(payload)
        gdb.write(f"wrote intermediate seam masks for stage {self.stage}\n")
        self.stage += 1
        if self.stage == 4:
            self.enabled = False
        return False


class SeamPairsBegin(gdb.Breakpoint):
    def stop(self):
        rsp = int(gdb.parse_and_eval("$rsp"))
        begin = read_u64(rsp + 0x30)
        end = read_u64(rsp + 0x38)
        count = (end - begin) // 96 if end >= begin else 0
        if count != 4:
            raise gdb.GdbError("unexpected initial seam mask vector layout")
        for index in range(count):
            mat = begin + index * 96
            rows = read_i32(mat + 8)
            cols = read_i32(mat + 12)
            data = read_u64(mat + 16)
            step_pointer = read_u64(mat + 72)
            row_step = read_u64(step_pointer)
            payload = bytearray(rows * cols)
            for row in range(rows):
                source = bytes(inferior.read_memory(data + row * row_step, cols))
                payload[row * cols:(row + 1) * cols] = source
            path = f"/tmp/navvis-original-seam-initial-cam{index}.pgm"
            with open(path, "wb") as output:
                output.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
                output.write(payload)
        gdb.write("wrote initial seam masks\n")
        self.enabled = False
        return False


# Return point of the recovered four-pair seam loop in build
# 05ad952633e1dadaf52fb40c2187932e20f8a5ee.
SeamMasksReady(f"*0x{base + 0x1afda8:x}")
SeamPairEntry(f"*0x{base + 0x1a9600:x}")
SeamPairReturn(f"*0x{base + 0x1afb60:x}")
SeamPairsBegin(f"*0x{base + 0x1afb5b:x}")
end

continue
