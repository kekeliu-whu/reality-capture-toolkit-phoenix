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
output_dir = os.environ.get(
    "NAVVIS_GRAPHCUT_IMAGE_DIR",
    "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/"
    "work/panorama_alignment_20260827/graphcut_image_capture",
)
os.makedirs(output_dir, exist_ok=True)


def words(address, size):
    raw = bytes(inferior.read_memory(address, size))
    return struct.unpack("<" + "Q" * (size // 8), raw)


def umat_payload(umat_address, channels):
    umat = words(umat_address, 80)
    flags = umat[0] & 0xffffffff
    rows = umat[1] & 0xffffffff
    cols = umat[1] >> 32
    umat_data = umat[4]
    offset = umat[5]
    step0 = umat[8]
    if not umat_data or rows == 0 or cols == 0 or step0 < cols * channels:
        raise RuntimeError(
            f"invalid UMat rows={rows} cols={cols} step={step0} data={umat_data:#x}"
        )
    storage = words(umat_data, 48)
    data = storage[3] + offset
    raw = bytes(inferior.read_memory(data, rows * step0))
    payload = b"".join(
        raw[row * step0:row * step0 + cols * channels] for row in range(rows)
    )
    return flags, rows, cols, payload


def dump_vector(vector_address, serial, kind, channels):
    begin, end, _ = words(vector_address, 24)
    if not begin or end < begin or (end - begin) % 80:
        gdb.write(
            f"GRAPHCUT_{kind.upper()}_VECTOR_INVALID serial={serial} "
            f"begin={begin:#x} end={end:#x}\n"
        )
        return
    count = (end - begin) // 80
    for index in range(count):
        flags, rows, cols, payload = umat_payload(begin + index * 80, channels)
        digest = hashlib.sha256(payload).hexdigest()
        path = os.path.join(
            output_dir, f"pair{serial}_{kind}{index}_{cols}x{rows}.raw"
        )
        with open(path, "wb") as output:
            output.write(payload)
        gdb.write(
            f"GRAPHCUT_{kind.upper()} serial={serial} index={index} "
            f"flags={flags:#x} size={cols}x{rows} sha256={digest} path={path}\n"
        )


class GraphCutEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        serial = type(self).serial
        images = int(gdb.parse_and_eval("$rsi"))
        masks = int(gdb.parse_and_eval("$rcx"))
        # GraphCut receives CV_32FC3 UMat images (12 bytes per pixel).
        dump_vector(images, serial, "image", 12)
        dump_vector(masks, serial, "mask", 1)
        return False


GraphCutEntry(
    "_ZN2cv6detail18GraphCutSeamFinder4findERKSt6vectorINS_4UMatESaIS3_EE"
    "RKS2_INS_6Point_IiEESaIS9_EERS5_"
)
end

continue
