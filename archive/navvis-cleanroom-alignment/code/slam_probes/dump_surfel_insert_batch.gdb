set pagination off
set confirm off
starti
python
import gdb
import struct

OUTPUT = "/tmp/navvis_vendor_surfel_insert_batch.bin"
IMAGE_BASE = 0x555555554000

class SurfelInsertBatch(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x48ed50), internal=True)

    def stop(self):
        worker = int(gdb.parse_and_eval("$rdi"))
        vector_pointer = struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(worker, 8))
        )[0]
        begin, end = struct.unpack(
            "<QQ", bytes(gdb.selected_inferior().read_memory(vector_pointer, 16))
        )
        count = (end - begin) // 24
        rays = bytes(gdb.selected_inferior().read_memory(begin, count * 24))
        with open(OUTPUT, "wb") as stream:
            stream.write(b"NVSURF1\0")
            stream.write(struct.pack("<Q", count))
            stream.write(rays)
        gdb.write(
            "SURFEL_BATCH worker=%#x vector=%#x begin=%#x end=%#x count=%d\n"
            % (worker, vector_pointer, begin, end, count)
        )
        gdb.execute("quit")
        return False

SurfelInsertBatch()
end
continue
