set pagination off
set confirm off
set logging enabled off
set print thread-events off
starti

python
import gdb
import struct


IMAGE_BASE = 0x555555554000
REMOVE_TILT_READY = IMAGE_BASE + 0x48C4E9
ADD_SUBMAP = IMAGE_BASE + 0x6084F0


def values(address, count):
    raw = gdb.selected_inferior().read_memory(address, 8 * count).tobytes()
    return struct.unpack(f"<{count}d", raw)


def format_values(items):
    return ",".join(format(value, ".17g") for value in items)


latest_remove_tilt = None


class RemoveTiltReady(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{REMOVE_TILT_READY:#x}", internal=True)

    def stop(self):
        global latest_remove_tilt
        stack = int(gdb.parse_and_eval("$rsp"))
        latest_remove_tilt = values(stack, 4)
        return False


class AddSubmap(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{ADD_SUBMAP:#x}", internal=True)
        self.count = 0

    def stop(self):
        self.count += 1
        pose = int(gdb.parse_and_eval("$rsi"))
        tilt = "unavailable" if latest_remove_tilt is None else format_values(
            latest_remove_tilt
        )
        gdb.write(
            f"SUBMAP_REMOVE_TILT count={self.count} values={tilt} "
            f"pose={format_values(values(pose, 8))}\n"
        )
        if self.count == 3:
            gdb.execute("quit")
        return False


RemoveTiltReady()
AddSubmap()
end

continue
