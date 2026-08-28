set pagination off
set confirm off
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_online_stage1_object.log
set logging overwrite on
set logging enabled on
starti

python
import gdb
import struct


IMAGE_BASE = 0x555555554000
SOLVE = IMAGE_BASE + 0x5A70D0


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{SOLVE:#x}", internal=False)

    def stop(self):
        inferior = gdb.selected_inferior()
        owner = int(gdb.parse_and_eval("$rsi"))
        raw = inferior.read_memory(owner, 0x300).tobytes()
        words = struct.unpack("<96Q", raw)
        gdb.write(f"SOLVE_OBJECT address={owner:#x}\n")
        for index, value in enumerate(words):
            gdb.write(f"OBJECT offset={8 * index:#x} value={value:#x}\n")
            try:
                child = inferior.read_memory(value, 0x100).tobytes()
            except gdb.MemoryError:
                continue
            child_words = struct.unpack("<32Q", child)
            gdb.write(
                f"CHILD object_offset={8 * index:#x} address={value:#x} "
                + " ".join(f"{word:016x}" for word in child_words)
                + "\n"
            )
        gdb.execute("set logging enabled off")
        gdb.execute("quit")
        return False


SolveEntry()
end

continue
