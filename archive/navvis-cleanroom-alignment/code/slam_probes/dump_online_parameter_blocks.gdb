set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
import struct


IMAGE_BASE = 0x555555554000
CERES_SOLVE = IMAGE_BASE + 0xEDD070


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{CERES_SOLVE:#x}", internal=True)
        self.call = 0

    def stop(self):
        self.call += 1
        if self.call != 1:
            return False
        problem = int(gdb.parse_and_eval("$rsi"))
        implementation = u64(problem)
        program = u64(implementation + 160)
        begin = u64(program)
        end = u64(program + 8)
        if end < begin or (end - begin) % 8:
            raise gdb.GdbError("invalid Ceres parameter-block vector")
        pointers = struct.unpack(f"<{(end - begin) // 8}Q", read(begin, end - begin))
        gdb.write(
            f"PARAMETER_VECTOR call=1 problem={problem:#x} program={program:#x} "
            f"count={len(pointers)}\n"
        )
        for index, pointer in enumerate(pointers):
            raw = read(pointer, 128)
            words = struct.unpack("<16Q", raw)
            gdb.write(
                f"PARAMETER_OBJECT index={index} address={pointer:#x} "
                + " ".join(f"{word:#x}" for word in words)
                + "\n"
            )
            for offset in range(0, 64, 8):
                candidate = words[offset // 8]
                try:
                    values = struct.unpack("<8d", read(candidate, 64))
                except gdb.MemoryError:
                    continue
                gdb.write(
                    f"PARAMETER_STORAGE index={index} object_offset={offset} "
                    f"address={candidate:#x} "
                    + " ".join(format(value, ".17g") for value in values)
                    + "\n"
                )
        gdb.execute("quit")
        return False


SolveEntry()
end

continue
