set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
import hashlib
import os
import struct


inferior = gdb.selected_inferior()
output_directory = os.environ.get("NAVVIS_WORLD_MAP_RESIZE_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_WORLD_MAP_RESIZE_DUMP_DIR is required")
os.makedirs(output_directory, exist_ok=True)

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


def words(address, size):
    raw = bytes(inferior.read_memory(address, size))
    return struct.unpack("<" + "Q" * (size // 8), raw)


def input_array_mat(input_array_address):
    return words(input_array_address, 16)[1]


def mat_metadata(mat_address):
    values = words(mat_address, 96)
    flags = values[0] & 0xFFFFFFFF
    rows = values[1] & 0xFFFFFFFF
    cols = values[1] >> 32
    return flags & 0xFFF, rows, cols, values[2], values[10]


def dump_float_mat(mat_address, filename_stem, label):
    mat_type, rows, cols, data, row_step = mat_metadata(mat_address)
    depth = mat_type & 7
    channels = (mat_type >> 3) + 1
    row_bytes = cols * channels * 4
    if depth != 5 or not rows or not cols or row_step < row_bytes:
        gdb.write(
            f"{label}: unsupported Mat at 0x{mat_address:x}: "
            f"type={mat_type} rows={rows} cols={cols} step={row_step}\n"
        )
        return
    raw = bytes(inferior.read_memory(data, rows * row_step))
    payload = b"".join(
        raw[row * row_step:row * row_step + row_bytes] for row in range(rows)
    )
    path = os.path.join(output_directory, f"{filename_stem}.f32c{channels}")
    with open(path, "wb") as output:
        output.write(payload)
    gdb.write(
        f"{label}: type={mat_type} channels={channels} {cols}x{rows} "
        f"bytes={len(payload)} "
        f"sha256={hashlib.sha256(payload).hexdigest()} {path}\n"
    )


class ResizeReturn(gdb.Breakpoint):
    def __init__(self, address, output_mat, serial, entry_breakpoint):
        super().__init__(f"*0x{address:x}", temporary=True, internal=True)
        self.output_mat = output_mat
        self.serial = serial
        self.entry_breakpoint = entry_breakpoint

    def stop(self):
        dump_float_mat(
            self.output_mat,
            f"projection_resize_{self.serial:02d}_output",
            f"PROJECTION_RESIZE_OUTPUT serial={self.serial}",
        )
        self.entry_breakpoint.enabled = True
        gdb.execute("set scheduler-locking off")
        return False


class ResizeEntry(gdb.Breakpoint):
    def __init__(self, address):
        super().__init__(f"*0x{address:x}")
        self.serial = 0

    def stop(self):
        source_mat = input_array_mat(int(gdb.parse_and_eval("$rdi")))
        output_mat = input_array_mat(int(gdb.parse_and_eval("$rsi")))
        mat_type, rows, cols, _, _ = mat_metadata(source_mat)
        size_address = int(gdb.parse_and_eval("$rdx"))
        width, height = struct.unpack("<ii", inferior.read_memory(size_address, 8))
        interpolation = int(gdb.parse_and_eval("$ecx")) & 0xFFFFFFFF
        caller = int(gdb.parse_and_eval("*(unsigned long long*)$rsp"))
        if (
            mat_type not in (5, 13, 21)
            or rows != 1024
            or cols != 2048
            or width != 8192
            or height != 4096
        ):
            return False

        serial = self.serial
        self.serial += 1
        gdb.write(
            f"PROJECTION_RESIZE_ENTRY serial={serial} type={mat_type} "
            f"caller_offset=0x{caller - pie_base:x} "
            f"source={cols}x{rows} target={width}x{height} "
            f"interpolation={interpolation}\n"
        )
        dump_float_mat(
            source_mat,
            f"projection_resize_{serial:02d}_input",
            f"PROJECTION_RESIZE_INPUT serial={serial}",
        )
        gdb.execute("set scheduler-locking on")
        self.enabled = False
        ResizeReturn(caller, output_mat, serial, self)
        return False


ResizeEntry(pie_base + 0x65F50)
end

continue
