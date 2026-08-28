set pagination off
set confirm off
set logging enabled off
set print thread-events off
starti

python
import gdb
import struct


IMAGE_BASE = 0x555555554000
SUBMAP_UPDATE = IMAGE_BASE + 0x608E80


def doubles(address, count):
    try:
        raw = gdb.selected_inferior().read_memory(address, 8 * count).tobytes()
        return ",".join(format(value, ".17g") for value in struct.unpack(f"<{count}d", raw))
    except gdb.MemoryError:
        return "unreadable"


class SubmapUpdate(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{SUBMAP_UPDATE:#x}", internal=True)
        self.count = 0

    def stop(self):
        self.count += 1
        registers = {
            name: int(gdb.parse_and_eval(f"${name}"))
            for name in ("rsi", "rdx", "rcx", "r8", "r9")
        }
        gdb.write(f"SUBMAP_UPDATE count={self.count}\n")
        for name, address in registers.items():
            gdb.write(
                f"SUBMAP_INPUT register={name} address={address:#x} "
                f"values={doubles(address, 8)}\n"
            )
        if self.count == 3:
            gdb.execute("quit")
        return False


SubmapUpdate()
end

continue
