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

output_directory = os.environ.get("NAVVIS_BLEND_MASK_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_BLEND_MASK_DUMP_DIR is required")
os.makedirs(output_directory, exist_ok=True)


def words(address, size=96):
    raw = bytes(inferior.read_memory(address, size))
    return struct.unpack("<" + "Q" * (size // 8), raw)


def vector_elements(address):
    begin, end, _ = words(address, 24)
    if not begin or end < begin or (end - begin) % 32:
        raise gdb.GdbError(f"invalid panorama vector at 0x{address:x}")
    return [begin + index * 32 for index in range((end - begin) // 32)]


def first_mat(panorama_address):
    begin, end, _ = words(panorama_address, 24)
    if not begin or end - begin < 96:
        raise gdb.GdbError(f"panorama at 0x{panorama_address:x} has no cv::Mat")
    return begin


def compact_mask(mat_address):
    values = words(mat_address, 96)
    flags = values[0] & 0xFFFFFFFF
    rows = values[1] & 0xFFFFFFFF
    cols = values[1] >> 32
    data = values[2]
    row_step = values[10]
    if (flags & 0xFFF) != 0 or row_step < cols:
        raise gdb.GdbError(f"unexpected mask at 0x{mat_address:x}")
    raw = bytes(inferior.read_memory(data, rows * row_step))
    payload = b"".join(
        raw[row * row_step:row * row_step + cols] for row in range(rows)
    )
    return rows, cols, payload


def dump_masks(vector_address, blend_serial, argument_index):
    for camera, panorama_address in enumerate(vector_elements(vector_address)):
        rows, cols, payload = compact_mask(first_mat(panorama_address))
        path = os.path.join(
            output_directory,
            f"blend{blend_serial}_arg{argument_index}_cam{camera}.pgm",
        )
        with open(path, "wb") as output:
            output.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
            output.write(payload)
        gdb.write(
            f"BLEND_MASK blend={blend_serial} arg={argument_index} cam={camera} "
            f"size={cols}x{rows} nonzero={sum(value != 0 for value in payload)} "
            f"sha256={hashlib.sha256(payload).hexdigest()}\n"
        )


class BlendEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        serial = type(self).serial
        dump_masks(int(gdb.parse_and_eval("$rcx")), serial, 1)
        dump_masks(int(gdb.parse_and_eval("$r8")), serial, 2)
        return False


BlendEntry(f"*0x{pie_base + 0x1e7ba0:x}")
end

continue
