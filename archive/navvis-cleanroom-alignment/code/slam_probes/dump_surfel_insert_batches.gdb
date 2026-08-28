set pagination off
set confirm off
starti
python
import gdb
import hashlib
import struct

OUTPUT = "/tmp/navvis_vendor_surfel_insert_batches.bin"
IMAGE_BASE = 0x555555554000

with open(OUTPUT, "wb") as stream:
    stream.write(b"NVSURFS1")

class SurfelInsertBatches(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x48ed50), internal=True)
        self.seen = set()
        self.count = 0

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
        digest = hashlib.sha256(rays).digest()
        if digest in self.seen:
            return False
        self.seen.add(digest)
        with open(OUTPUT, "ab") as stream:
            stream.write(struct.pack("<Q", count))
            stream.write(rays)
        gdb.write("SURFEL_BATCH index=%d count=%d sha256=%s\n" % (
            self.count, count, digest.hex()
        ))
        self.count += 1
        if self.count >= 3:
            gdb.execute("quit")
        return False

SurfelInsertBatches()
end
continue
