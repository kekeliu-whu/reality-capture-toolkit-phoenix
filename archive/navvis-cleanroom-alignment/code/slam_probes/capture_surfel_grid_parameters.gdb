set pagination off
set confirm off
starti
python
import gdb
import struct


GRID_INSERT_OFFSET = 0x499970


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


class FirstGridInsert(gdb.Breakpoint):
    def stop(self):
        grid = int(gdb.parse_and_eval("$rdi"))
        raw = bytes(gdb.selected_inferior().read_memory(grid + 0x20, 0x20))
        origin_x, origin_y, origin_z, inverse_resolution = struct.unpack("<4d", raw)
        options = bytes(gdb.selected_inferior().read_memory(grid + 0xc0, 0x18))
        gdb.write(
            "SURFEL_GRID grid=%#x origin=(%.17g, %.17g, %.17g) "
            "inverse_resolution=%.17g raw=%s options=%s\n"
            % (
                grid,
                origin_x,
                origin_y,
                origin_z,
                inverse_resolution,
                raw.hex(),
                options.hex(),
            )
        )
        gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
FirstGridInsert("*%#x" % (base + GRID_INSERT_OFFSET), internal=True)
end
continue
