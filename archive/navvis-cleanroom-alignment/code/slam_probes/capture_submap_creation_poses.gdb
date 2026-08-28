set pagination off
set confirm off
set logging enabled off
set print thread-events off
starti

python
import gdb
import struct


IMAGE_BASE = 0x555555554000
ADD_SUBMAP = IMAGE_BASE + 0x6084F0


class AddSubmap(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{ADD_SUBMAP:#x}", internal=True)
        self.count = 0

    def stop(self):
        self.count += 1
        pose = int(gdb.parse_and_eval("$rsi"))
        timestamp = int(gdb.parse_and_eval("$rdx"))
        values = struct.unpack(
            "<8d", gdb.selected_inferior().read_memory(pose, 64).tobytes()
        )
        gdb.write(
            f"SUBMAP_CREATE count={self.count} pose={pose:#x} "
            f"timestamp_pointer={timestamp:#x} values="
            + ",".join(format(value, ".17g") for value in values)
            + "\n"
        )
        if self.count == 3:
            gdb.execute("quit")
        return False


AddSubmap()
end

continue
