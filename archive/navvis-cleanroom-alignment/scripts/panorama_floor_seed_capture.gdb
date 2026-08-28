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
        start = int(fields[0].split("-", 1)[0], 16)
        file_offset = int(fields[2], 16)
        candidate = start - file_offset
        pie_base = candidate if pie_base is None else min(pie_base, candidate)

if pie_base is None:
    raise gdb.GdbError(f"could not locate PIE base for {executable}")

output_directory = os.environ.get("NAVVIS_FLOOR_SEED_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_FLOOR_SEED_DUMP_DIR is required")
os.makedirs(output_directory, exist_ok=True)


def mat_info(address):
    raw = bytes(inferior.read_memory(address, 96))
    values = struct.unpack("<12Q", raw)
    flags = values[0] & 0xFFFFFFFF
    rows = values[1] & 0xFFFFFFFF
    cols = values[1] >> 32
    return flags & 0xFFF, rows, cols, values[2], values[10]


def dump_mat(address, stem, expected_type):
    mat_type, rows, cols, data, row_step = mat_info(address)
    if mat_type != expected_type:
        raise gdb.GdbError(
            f"unexpected Mat type {mat_type} at 0x{address:x}; expected {expected_type}"
        )
    element_bytes = 12 if expected_type == 21 else 1
    row_bytes = cols * element_bytes
    payload = b"".join(
        bytes(inferior.read_memory(data + row * row_step, row_bytes))
        for row in range(rows)
    )
    suffix = "f32le" if expected_type == 21 else "u8"
    path = os.path.join(output_directory, f"{stem}.{suffix}")
    with open(path, "wb") as output:
        output.write(payload)
    gdb.write(
        f"DUMP {stem}={cols}x{rows}/type{mat_type} bytes={len(payload)} "
        f"sha256={hashlib.sha256(payload).hexdigest()} {path}\n"
    )


state = {"wrapper_serial": 0, "active": False, "down": 0, "up": 0}


class WrapperReturn(gdb.FinishBreakpoint):
    def stop(self):
        state["active"] = False
        return False


class WrapperEntry(gdb.Breakpoint):
    def stop(self):
        state["wrapper_serial"] += 1
        if state["wrapper_serial"] != 9:
            return False
        state["active"] = True
        state["down"] = 0
        state["up"] = 0
        WrapperReturn(gdb.newest_frame(), internal=True)
        gdb.write("FLOOR_SEED_CAPTURE_ACTIVE\n")
        return False


class DownReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, image, mask, serial):
        super().__init__(frame, internal=True)
        self.image = image
        self.mask = mask
        self.serial = serial

    def stop(self):
        dump_mat(self.image, f"vendor_down_{self.serial}_image", 21)
        dump_mat(self.mask, f"vendor_down_{self.serial}_mask", 0)
        return False


class DownEntry(gdb.Breakpoint):
    def stop(self):
        if not state["active"]:
            return False
        state["down"] += 1
        if state["down"] not in (10, 11):
            return False
        destination = int(gdb.parse_and_eval("$rdx"))
        destination_mask = int(gdb.parse_and_eval("$rcx"))
        DownReturn(
            gdb.newest_frame(), destination, destination_mask, state["down"]
        )
        return False


class UpEntry(gdb.Breakpoint):
    def stop(self):
        if not state["active"]:
            return False
        state["up"] += 1
        if state["up"] == 1:
            dump_mat(
                int(gdb.parse_and_eval("$rsi")), "vendor_first_up_coarse", 21
            )
        return False


WrapperEntry(f"*0x{pie_base + 0x22C2D0:x}")
DownEntry(f"*0x{pie_base + 0x229AB0:x}")
UpEntry(f"*0x{pie_base + 0x229560:x}")
end

continue
