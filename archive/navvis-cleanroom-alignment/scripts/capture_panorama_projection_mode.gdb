set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
import os


inferior = gdb.selected_inferior()
executable = os.path.realpath(gdb.current_progspace().filename)
pie_base = None
with open(f"/proc/{inferior.pid}/maps", "r", encoding="ascii") as maps_file:
    for line in maps_file:
        fields = line.split()
        if len(fields) < 6 or os.path.realpath(fields[-1]) != executable:
            continue
        mapping_start = int(fields[0].split("-", 1)[0], 16)
        file_offset = int(fields[2], 16)
        candidate = mapping_start - file_offset
        pie_base = candidate if pie_base is None else min(pie_base, candidate)

if pie_base is None:
    raise gdb.GdbError(f"could not locate PIE base for {executable}")


class ProjectionModeEntry(gdb.Breakpoint):
    def stop(self):
        rolling_shutter = int(gdb.parse_and_eval("$dl")) & 0xFF
        readout_seconds = float(gdb.parse_and_eval("$xmm0.v2_double[0]"))
        gdb.write(
            f"PROJECTION_MODE rolling-shutter={rolling_shutter} "
            f"readout-seconds={readout_seconds:.17g}\n"
        )
        for label, register in (
            ("linear-velocity", "$rcx"),
            ("angular-velocity", "$r8"),
            ("point-ccs", "$r9"),
        ):
            address = int(gdb.parse_and_eval(register))
            values = [
                float(gdb.parse_and_eval(f"*(double*)({address + offset})"))
                for offset in (0, 8, 16)
            ]
            gdb.write(f"{label}={values}\n")
        gdb.execute("quit")
        return False


ProjectionModeEntry(f"*0x{pie_base + 0x1A6030:x}")
end

continue
