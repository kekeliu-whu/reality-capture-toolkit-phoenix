set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
starti

python
import gdb
import hashlib
import os
import struct


inferior = gdb.selected_inferior()
output_directory = os.environ.get("NAVVIS_FINAL_IMWRITE_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_FINAL_IMWRITE_DUMP_DIR is required")
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


def read_string(address):
    data, length = words(address, 16)
    return bytes(inferior.read_memory(data, length)).decode("utf-8", errors="replace")


def dump_bgr_mat(mat_address, filename, label):
    values = words(mat_address, 96)
    flags = values[0] & 0xFFFFFFFF
    mat_type = flags & 0xFFF
    rows = values[1] & 0xFFFFFFFF
    cols = values[1] >> 32
    data = values[2]
    row_step = values[10]
    if mat_type != 16 or not rows or not cols or row_step < cols * 3:
        gdb.write(
            f"{label}: unsupported Mat at 0x{mat_address:x}: "
            f"type={mat_type} rows={rows} cols={cols} step={row_step}\n"
        )
        return
    raw = bytes(inferior.read_memory(data, rows * row_step))
    payload = b"".join(
        raw[row * row_step:row * row_step + cols * 3] for row in range(rows)
    )
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


class ImwriteEntry(gdb.Breakpoint):
    def stop(self):
        path = read_string(int(gdb.parse_and_eval("$rdi")))
        if not (
            path.endswith("/Images/00000-pano.jpg")
            or path.endswith("/ImagesFilled/00000-pano.jpg")
        ):
            return False
        input_array = int(gdb.parse_and_eval("$rsi"))
        input_words = words(input_array, 24)
        mat_address = input_words[1]
        label = "FINAL_IMWRITE" if "/ImagesFilled/" in path else "NO_FLOOR_IMWRITE"
        filename = "final_imwrite_input.tga" if "/ImagesFilled/" in path else "no_floor_imwrite_input.tga"
        gdb.write(
            f"{label}_ARRAY path={path} flags={input_words[0]:#x} "
            f"object={mat_address:#x} size-word={input_words[2]:#x}\n"
        )
        dump_bgr_mat(mat_address, filename, label)
        return False


# Both imported writers use the same first two ABI arguments.  The renderer
# uses navvis::io::imwrite for some artifacts and cv::imwrite for others.
ImwriteEntry(f"*0x{pie_base + 0x66040:x}")
ImwriteEntry(f"*0x{pie_base + 0x66330:x}")
end

continue
