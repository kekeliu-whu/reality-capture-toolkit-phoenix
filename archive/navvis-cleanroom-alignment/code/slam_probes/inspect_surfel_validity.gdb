set pagination off
set confirm off
starti
python
import gdb
import struct

IMAGE_BASE = 0x555555554000

class FirstSurfelValidity(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x4985d0), internal=True)

    def stop(self):
        thresholds = int(gdb.parse_and_eval("$rdi"))
        surfel = int(gdb.parse_and_eval("$rsi"))
        voxel_scale = float(gdb.parse_and_eval("$xmm0.v4_float[0]"))
        threshold_values = struct.unpack(
            "<6f", bytes(gdb.selected_inferior().read_memory(thresholds, 24))
        )
        surfel_bytes = bytes(gdb.selected_inferior().read_memory(surfel, 104))
        weight = struct.unpack_from("<f", surfel_bytes, 8)[0]
        count = struct.unpack_from("<I", surfel_bytes, 12)[0]
        center = struct.unpack_from("<3f", surfel_bytes, 16)
        eigenvalues = struct.unpack_from("<3f", surfel_bytes, 88)
        gdb.write(
            "SURFEL_VALIDITY scale=%r thresholds=%r weight=%r count=%r "
            "center=%r eigenvalues=%r\n"
            % (voxel_scale, threshold_values, weight, count, center, eigenvalues)
        )
        gdb.execute("bt 12")
        gdb.execute("quit")
        return False

FirstSurfelValidity()
end
continue
