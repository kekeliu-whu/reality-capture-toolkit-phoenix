set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on

starti

python
import gdb
import os
import struct

RAYCAST_CALL = 0x55555573694C
output_path = os.environ["PCT_SURFEL_OUTPUT"]


def u64(address):
    memory = gdb.selected_inferior().read_memory(address, 8)
    return struct.unpack("<Q", memory)[0]


class RaycastInputBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % RAYCAST_CALL, temporary=True, internal=False)

    def stop(self):
        raycaster = int(gdb.parse_and_eval("$rdi"))
        point_cloud = u64(raycaster + 0x18)
        begin = u64(point_cloud)
        end = u64(point_cloud + 8)
        byte_count = end - begin
        if byte_count <= 0 or byte_count % 32 != 0:
            raise RuntimeError(
                "unexpected PCT surfel vector: begin=0x%x end=0x%x" % (begin, end)
            )
        data = bytes(gdb.selected_inferior().read_memory(begin, byte_count))
        with open(output_path, "wb") as output:
            output.write(data)
        gdb.write(
            "PCT_SURFEL_CAPTURE path=%s begin=0x%x count=%d bytes=%d\n"
            % (output_path, begin, byte_count // 32, byte_count)
        )
        return True


RaycastInputBreakpoint()
end

continue
quit
