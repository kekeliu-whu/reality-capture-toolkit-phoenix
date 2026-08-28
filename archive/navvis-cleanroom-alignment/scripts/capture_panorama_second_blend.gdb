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

output_directory = os.environ.get("NAVVIS_SECOND_BLEND_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_SECOND_BLEND_DUMP_DIR is required")
os.makedirs(output_directory, exist_ok=True)
gdb.write(f"renderer PIE base: 0x{pie_base:x}\n")
gdb.write(f"second-blend dump: {output_directory}\n")


def words(address, size=96):
    raw = bytes(inferior.read_memory(address, size))
    return struct.unpack("<" + "Q" * (size // 8), raw)


def vector_elements(address):
    begin, end, _ = words(address, 24)
    if not begin or end < begin or (end - begin) % 32 != 0:
        raise gdb.GdbError(
            f"invalid panorama vector at 0x{address:x}: [{begin:#x}, {end:#x})"
        )
    return [begin + index * 32 for index in range((end - begin) // 32)]


def first_mat(panorama_address):
    begin, end, _ = words(panorama_address, 24)
    if not begin or end - begin < 96:
        raise gdb.GdbError(f"panorama at 0x{panorama_address:x} has no cv::Mat")
    return begin


def mat_metadata(mat_address):
    values = words(mat_address, 96)
    flags = values[0] & 0xFFFFFFFF
    rows = values[1] & 0xFFFFFFFF
    cols = values[1] >> 32
    return flags & 0xFFF, rows, cols, values[2], values[10]


def compact_mat_payload(mat_address, expected_type):
    mat_type, rows, cols, data, row_step = mat_metadata(mat_address)
    if mat_type != expected_type:
        raise gdb.GdbError(
            f"unexpected Mat type {mat_type} at 0x{mat_address:x}; expected {expected_type}"
        )
    channels = 3 if expected_type == 16 else 1
    row_bytes = cols * channels
    if row_step < row_bytes:
        raise gdb.GdbError(
            f"invalid Mat row step {row_step} for {cols} columns and {channels} channels"
        )
    raw = bytes(inferior.read_memory(data, rows * row_step))
    payload = b"".join(
        raw[row * row_step:row * row_step + row_bytes] for row in range(rows)
    )
    return rows, cols, payload


def dump_bgr_panorama(panorama_address, filename, label):
    dump_bgr_mat(first_mat(panorama_address), filename, label)


def dump_bgr_mat(mat_address, filename, label):
    rows, cols, payload = compact_mat_payload(mat_address, 16)
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


def dump_mask_vector(vector_address, argument_index):
    for camera, panorama_address in enumerate(vector_elements(vector_address)):
        rows, cols, payload = compact_mat_payload(first_mat(panorama_address), 0)
        path = os.path.join(
            output_directory, f"blend2_arg{argument_index}_cam{camera}.pgm"
        )
        with open(path, "wb") as output:
            output.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
            output.write(payload)
        gdb.write(
            f"BLEND2_MASK arg={argument_index} cam={camera}: {cols}x{rows} "
            f"nonzero={sum(value != 0 for value in payload)} "
            f"full={sum(value == 255 for value in payload)} "
            f"sha256={hashlib.sha256(payload).hexdigest()} {path}\n"
        )


class BlendReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, hidden_return):
        super().__init__(frame, internal=True)
        self.hidden_return = hidden_return

    def stop(self):
        dump_bgr_panorama(
            self.hidden_return, "blend2_output.tga", "BLEND2_OUTPUT"
        )
        return False


class BlendEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        if type(self).serial != 2:
            return False
        hidden_return = int(gdb.parse_and_eval("$rdi"))
        blender = int(gdb.parse_and_eval("$rsi"))
        image_vector = int(gdb.parse_and_eval("$rdx"))
        seam_mask_vector = int(gdb.parse_and_eval("$rcx"))
        projection_mask_vector = int(gdb.parse_and_eval("$r8"))
        blender_words = words(blender, 24)
        number_of_bands = blender_words[1] >> 32
        gdb.write(
            f"BLEND2_ENTRY hidden-return={hidden_return:#x} blender={blender:#x} "
            f"number-of-bands={number_of_bands} expected-levels={number_of_bands + 1}\n"
        )
        for camera, panorama_address in enumerate(vector_elements(image_vector)):
            dump_bgr_panorama(
                panorama_address,
                f"blend2_input_cam{camera}.tga",
                f"BLEND2_INPUT cam={camera}",
            )
        dump_mask_vector(seam_mask_vector, 1)
        dump_mask_vector(projection_mask_vector, 2)
        BlendReturn(gdb.newest_frame(), hidden_return)
        return False


class PyramidReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, destination, serial):
        super().__init__(frame, internal=True)
        self.destination = destination
        self.serial = serial

    def stop(self):
        if self.serial != 2:
            return False
        image_begin, image_end, _ = words(self.destination + 24, 24)
        weight_begin, weight_end, _ = words(self.destination + 48, 24)
        image_levels = (image_end - image_begin) // 32
        weight_levels = (weight_end - weight_begin) // 32
        gdb.write(
            f"BLEND2_PYRAMIDS image-levels={image_levels} "
            f"weight-levels={weight_levels}\n"
        )
        return False


class PyramidEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        PyramidReturn(
            gdb.newest_frame(), int(gdb.parse_and_eval("$rdi")), type(self).serial
        )
        return False


class SeamPreparedReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, serial, hidden_return):
        super().__init__(frame, internal=True)
        self.serial = serial
        self.hidden_return = hidden_return

    def stop(self):
        if 5 <= self.serial <= 8:
            camera = self.serial - 5
            dump_bgr_mat(
                self.hidden_return,
                f"blend2_seam_prepared_cam{camera}.tga",
                f"BLEND2_SEAM_PREPARED cam={camera}",
            )
        return False


class SeamPreparedEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        serial = type(self).serial
        if 5 <= serial <= 8:
            SeamPreparedReturn(
                gdb.newest_frame(), serial, int(gdb.parse_and_eval("$rdi"))
            )
        return False


BlendEntry(f"*0x{pie_base + 0x1E7BA0:x}")
PyramidEntry(f"*0x{pie_base + 0x1E6EE0:x}")
SeamPreparedEntry(f"*0x{pie_base + 0x22C2D0:x}")
end

continue
