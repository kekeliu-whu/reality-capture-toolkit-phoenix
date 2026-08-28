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

output_directory = os.environ.get("NAVVIS_FINAL_STAGE_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_FINAL_STAGE_DUMP_DIR is required")
os.makedirs(output_directory, exist_ok=True)
gdb.write(f"FINAL_STAGE_PIE base=0x{pie_base:x}\n")
gdb.write(f"FINAL_STAGE_DUMP directory={output_directory}\n")


def read_memory(address, size):
    return bytes(inferior.read_memory(address, size))


def words(address, size=96):
    raw = read_memory(address, size)
    return struct.unpack("<" + "Q" * (size // 8), raw)


def read_string(address):
    data, length = words(address, 16)
    if length > 1 << 20:
        return f"<invalid-string-length:{length}>"
    return read_memory(data, length).decode("utf-8", errors="replace")


def unwrap_array(array_address):
    header = words(array_address, 24)
    return header[0] & 0xFFFFFFFF, header[1]


def mat_metadata(mat_address):
    values = words(mat_address, 96)
    return {
        "type": values[0] & 0xFFF,
        "rows": values[1] & 0xFFFFFFFF,
        "cols": values[1] >> 32,
        "data": values[2],
        "step": values[10],
    }


def compact_mat_payload(mat_address, expected_type=None):
    meta = mat_metadata(mat_address)
    if expected_type is not None and meta["type"] != expected_type:
        raise gdb.GdbError(
            f"unexpected Mat type {meta['type']} at 0x{mat_address:x}; "
            f"expected {expected_type}"
        )
    channels = 3 if meta["type"] == 16 else 1
    item_size = 4 if meta["type"] in (5, 21) else 1
    row_bytes = meta["cols"] * channels * item_size
    if not meta["data"] or meta["step"] < row_bytes:
        raise gdb.GdbError(
            f"invalid Mat at 0x{mat_address:x}: {meta} row-bytes={row_bytes}"
        )
    raw = read_memory(meta["data"], meta["rows"] * meta["step"])
    payload = b"".join(
        raw[row * meta["step"]:row * meta["step"] + row_bytes]
        for row in range(meta["rows"])
    )
    return meta, payload


def dump_mask(mat_address, filename, label):
    meta, payload = compact_mat_payload(mat_address, 0)
    path = os.path.join(output_directory, filename)
    with open(path, "wb") as output:
        output.write(
            f"P5\n{meta['cols']} {meta['rows']}\n255\n".encode("ascii")
        )
        output.write(payload)
    gdb.write(
        f"{label} cols={meta['cols']} rows={meta['rows']} "
        f"nonzero={sum(value != 0 for value in payload)} "
        f"sha256={hashlib.sha256(payload).hexdigest()} path={path}\n"
    )


def vector_ints(vector_address):
    begin, end, _ = words(vector_address, 24)
    if not begin and not end:
        return []
    if end < begin or (end - begin) % 4:
        return [f"invalid:{begin:#x}:{end:#x}"]
    count = (end - begin) // 4
    if count > 64:
        return [f"too-many:{count}"]
    return list(struct.unpack("<" + "i" * count, read_memory(begin, count * 4)))


class BitwiseOrReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, serial, output_mat):
        super().__init__(frame, internal=True)
        self.serial = serial
        self.output_mat = output_mat

    def stop(self):
        try:
            meta = mat_metadata(self.output_mat)
            if meta["type"] == 0 and meta["rows"] == 4096 and meta["cols"] == 8192:
                dump_mask(
                    self.output_mat,
                    f"or_{self.serial:03d}_output.pgm",
                    f"MASK_OR_RETURN serial={self.serial}",
                )
        except gdb.error as error:
            gdb.write(f"MASK_OR_RETURN_ERROR serial={self.serial} error={error}\n")
        return False


class BitwiseOrEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        serial = type(self).serial
        try:
            _, input_a = unwrap_array(int(gdb.parse_and_eval("$rdi")))
            _, input_b = unwrap_array(int(gdb.parse_and_eval("$rsi")))
            _, output = unwrap_array(int(gdb.parse_and_eval("$rdx")))
            meta_a = mat_metadata(input_a)
            meta_b = mat_metadata(input_b)
            if (
                meta_a["type"] == 0
                and meta_b["type"] == 0
                and meta_a["rows"] == 4096
                and meta_a["cols"] == 8192
                and meta_b["rows"] == 4096
                and meta_b["cols"] == 8192
            ):
                dump_mask(input_a, f"or_{serial:03d}_input_a.pgm", f"MASK_OR_A serial={serial}")
                dump_mask(input_b, f"or_{serial:03d}_input_b.pgm", f"MASK_OR_B serial={serial}")
                BitwiseOrReturn(gdb.newest_frame(), serial, output)
        except gdb.error as error:
            gdb.write(f"MASK_OR_ENTRY_ERROR serial={serial} error={error}\n")
        return False


class ImreadReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, serial, hidden_return, path, flags):
        super().__init__(frame, internal=True)
        self.serial = serial
        self.hidden_return = hidden_return
        self.path = path
        self.flags = flags

    def stop(self):
        try:
            meta = mat_metadata(self.hidden_return)
            if meta["rows"] == 4096 and meta["cols"] == 8192:
                gdb.write(
                    f"IMREAD_RETURN serial={self.serial} path={self.path} "
                    f"flags={self.flags} type={meta['type']} cols={meta['cols']} "
                    f"rows={meta['rows']}\n"
                )
        except gdb.error as error:
            gdb.write(f"IMREAD_RETURN_ERROR serial={self.serial} error={error}\n")
        return False


class ImreadEntry(gdb.Breakpoint):
    serial = 0

    def stop(self):
        type(self).serial += 1
        serial = type(self).serial
        try:
            hidden_return = int(gdb.parse_and_eval("$rdi"))
            path = read_string(int(gdb.parse_and_eval("$rsi")))
            flags = int(gdb.parse_and_eval("$edx"))
            if "pano" in path or "/Images/" in path:
                gdb.write(
                    f"IMREAD_ENTRY serial={serial} path={path} flags={flags}\n"
                )
                ImreadReturn(
                    gdb.newest_frame(), serial, hidden_return, path, flags
                )
        except gdb.error as error:
            gdb.write(f"IMREAD_ENTRY_ERROR serial={serial} error={error}\n")
        return False


class ImwriteEntry(gdb.Breakpoint):
    serial = 0

    def __init__(self, specification, writer):
        super().__init__(specification)
        self.writer = writer

    def stop(self):
        type(self).serial += 1
        serial = type(self).serial
        try:
            path = read_string(int(gdb.parse_and_eval("$rdi")))
            array_flags, mat_address = unwrap_array(int(gdb.parse_and_eval("$rsi")))
            params = vector_ints(int(gdb.parse_and_eval("$rdx")))
            meta = mat_metadata(mat_address)
            if "pano" in path or meta["rows"] == 4096 or meta["cols"] == 8192:
                gdb.write(
                    f"IMWRITE_ENTRY serial={serial} writer={self.writer} path={path} "
                    f"array-flags=0x{array_flags:x} type={meta['type']} "
                    f"cols={meta['cols']} rows={meta['rows']} params={params}\n"
                )
        except gdb.error as error:
            gdb.write(
                f"IMWRITE_ENTRY_ERROR serial={serial} writer={self.writer} error={error}\n"
            )
        return False


BitwiseOrEntry(f"*0x{pie_base + 0x66600:x}")
ImreadEntry(f"*0x{pie_base + 0x65F70:x}")
ImwriteEntry(f"*0x{pie_base + 0x66040:x}", "navvis::io::imwrite")
ImwriteEntry(f"*0x{pie_base + 0x66330:x}", "cv::imwrite")
end

continue
