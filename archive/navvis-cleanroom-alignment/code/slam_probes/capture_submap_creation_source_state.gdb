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
ADD_SUBMAP = IMAGE_BASE + 0x6084F0


def read_doubles(address, count):
    raw = gdb.selected_inferior().read_memory(address, 8 * count).tobytes()
    return struct.unpack(f"<{count}d", raw)


def format_values(values):
    return ",".join(format(value, ".17g") for value in values)


latest = {}


class SubmapUpdate(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{SUBMAP_UPDATE:#x}", internal=True)

    def stop(self):
        for name in ("rcx", "r8", "r9"):
            address = int(gdb.parse_and_eval(f"${name}"))
            latest[name] = read_doubles(address, 8)
        return False


class AddSubmap(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{ADD_SUBMAP:#x}", internal=True)
        self.count = 0

    def stop(self):
        self.count += 1
        pose = int(gdb.parse_and_eval("$rsi"))
        gdb.write(
            f"SUBMAP_CREATE_SOURCE count={self.count} "
            f"pose={format_values(read_doubles(pose, 8))}\n"
        )
        for name in ("rcx", "r8", "r9"):
            if name not in latest:
                gdb.write(
                    f"SUBMAP_CREATE_LATEST register={name} values=unavailable\n"
                )
                continue
            gdb.write(
                f"SUBMAP_CREATE_LATEST register={name} "
                f"values={format_values(latest[name])}\n"
            )
        if self.count == 3:
            gdb.execute("quit")
        return False


SubmapUpdate()
AddSubmap()
end

continue
