set pagination off
set confirm off
starti
python
import gdb
import struct

IMAGE_BASE = 0x555555554000

class FirstSurfelInsert(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x499970), internal=True)

    def stop(self):
        ray = int(gdb.parse_and_eval("$rsi"))
        data = bytes(gdb.selected_inferior().read_memory(ray, 28))
        values = struct.unpack("<7f", data)
        gdb.write("SURFEL_INSERT ray=%#x values=%r\n" % (ray, values))
        gdb.execute("bt 20")
        gdb.execute("info registers rdi rsi rdx rcx r8 r9 rip")
        gdb.execute("quit")
        return False

FirstSurfelInsert()
end
continue
