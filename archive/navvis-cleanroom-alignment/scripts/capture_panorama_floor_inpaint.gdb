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

output_directory = os.environ.get("NAVVIS_FLOOR_INPAINT_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_FLOOR_INPAINT_DUMP_DIR is required")
os.makedirs(output_directory, exist_ok=True)


def words(address, size=96):
    raw = bytes(inferior.read_memory(address, size))
    return struct.unpack("<" + "Q" * (size // 8), raw)


def mat_payload(mat_address, expected_type):
    values = words(mat_address, 96)
    flags = values[0] & 0xFFFFFFFF
    mat_type = flags & 0xFFF
    rows = values[1] & 0xFFFFFFFF
    cols = values[1] >> 32
    data = values[2]
    row_step = values[10]
    if mat_type != expected_type:
        raise gdb.GdbError(
            f"unexpected Mat type {mat_type} at 0x{mat_address:x}; expected {expected_type}"
        )
    channels = 3 if expected_type == 16 else 1
    row_bytes = cols * channels
    raw = bytes(inferior.read_memory(data, rows * row_step))
    payload = b"".join(
        raw[row * row_step:row * row_step + row_bytes] for row in range(rows)
    )
    return rows, cols, payload


def dump_bgr(mat_address, filename, label):
    rows, cols, payload = mat_payload(mat_address, 16)
    header = struct.pack(
        "<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, cols, rows, 24, 0x20
    )
    path = os.path.join(output_directory, filename)
    with open(path, "wb") as output:
        output.write(header)
        output.write(payload)
    gdb.write(
        f"{label}: {cols}x{rows} bytes={len(payload)} "
        f"sha256={hashlib.sha256(payload).hexdigest()} {path}\n"
    )


def dump_mask(mat_address, filename, label):
    rows, cols, payload = mat_payload(mat_address, 0)
    path = os.path.join(output_directory, filename)
    with open(path, "wb") as output:
        output.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
        output.write(payload)
    gdb.write(
        f"{label}: {cols}x{rows} nonzero={sum(value != 0 for value in payload)} "
        f"full={sum(value == 255 for value in payload)} "
        f"sha256={hashlib.sha256(payload).hexdigest()} {path}\n"
    )


class FloorInpaintReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, hidden_return):
        super().__init__(frame, internal=True)
        self.hidden_return = hidden_return

    def stop(self):
        dump_bgr(self.hidden_return, "floor_inpaint_output.tga", "FLOOR_INPAINT_OUTPUT")
        return False


class PyramidInpaintEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        if type(self).serial != 9:
            return False
        hidden_return = int(gdb.parse_and_eval("$rdi"))
        image = int(gdb.parse_and_eval("$rsi"))
        mask = int(gdb.parse_and_eval("$rdx"))
        dump_bgr(image, "floor_inpaint_input.tga", "FLOOR_INPAINT_INPUT")
        dump_mask(mask, "floor_inpaint_mask.pgm", "FLOOR_INPAINT_MASK")
        FloorInpaintReturn(gdb.newest_frame(), hidden_return)
        return False


PyramidInpaintEntry(f"*0x{pie_base + 0x22C2D0:x}")
end

continue
