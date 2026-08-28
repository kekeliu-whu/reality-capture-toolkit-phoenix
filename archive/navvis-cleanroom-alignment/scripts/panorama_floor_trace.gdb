set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
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


def mat_info(address):
    raw = bytes(inferior.read_memory(address, 96))
    values = struct.unpack("<12Q", raw)
    flags = values[0] & 0xFFFFFFFF
    rows = values[1] & 0xFFFFFFFF
    cols = values[1] >> 32
    return flags & 0xFFF, rows, cols


state = {"serial": 0, "active": False, "down": 0, "up": 0}


class WrapperReturn(gdb.FinishBreakpoint):
    def stop(self):
        state["active"] = False
        gdb.write(
            f"FLOOR_RETURN down_calls={state['down']} up_calls={state['up']}\n"
        )
        return False


class WrapperEntry(gdb.Breakpoint):
    def stop(self):
        state["serial"] += 1
        if state["serial"] != 9:
            return False
        image = int(gdb.parse_and_eval("$rsi"))
        mask = int(gdb.parse_and_eval("$rdx"))
        image_type, rows, cols = mat_info(image)
        mask_type, mask_rows, mask_cols = mat_info(mask)
        state["active"] = True
        state["down"] = 0
        state["up"] = 0
        gdb.write(
            "FLOOR_ENTRY "
            f"image={cols}x{rows}/type{image_type} "
            f"mask={mask_cols}x{mask_rows}/type{mask_type} "
            f"ecx={int(gdb.parse_and_eval('$ecx'))} "
            f"r8d={int(gdb.parse_and_eval('$r8d'))}\n"
        )
        WrapperReturn(gdb.newest_frame(), internal=True)
        return False


class DownReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, image, mask, serial):
        super().__init__(frame, internal=True)
        self.image = image
        self.mask = mask
        self.serial = serial

    def stop(self):
        image_type, rows, cols = mat_info(self.image)
        mask_type, mask_rows, mask_cols = mat_info(self.mask)
        gdb.write(
            f"DOWN_RETURN {self.serial} image={cols}x{rows}/type{image_type} "
            f"mask={mask_cols}x{mask_rows}/type{mask_type}\n"
        )
        return False


class DownEntry(gdb.Breakpoint):
    def stop(self):
        if not state["active"]:
            return False
        state["down"] += 1
        source = int(gdb.parse_and_eval("$rdi"))
        source_mask = int(gdb.parse_and_eval("$rsi"))
        destination = int(gdb.parse_and_eval("$rdx"))
        destination_mask = int(gdb.parse_and_eval("$rcx"))
        image_type, rows, cols = mat_info(source)
        mask_type, mask_rows, mask_cols = mat_info(source_mask)
        serial = state["down"]
        gdb.write(
            f"DOWN_ENTRY {serial} source={cols}x{rows}/type{image_type} "
            f"mask={mask_cols}x{mask_rows}/type{mask_type} "
            f"r8d={int(gdb.parse_and_eval('$r8d'))}\n"
        )
        DownReturn(gdb.newest_frame(), destination, destination_mask, serial)
        return False


class UpEntry(gdb.Breakpoint):
    def stop(self):
        if not state["active"]:
            return False
        state["up"] += 1
        coarse = int(gdb.parse_and_eval("$rsi"))
        fine = int(gdb.parse_and_eval("$rdx"))
        fine_mask = int(gdb.parse_and_eval("$rcx"))
        coarse_type, coarse_rows, coarse_cols = mat_info(coarse)
        fine_type, fine_rows, fine_cols = mat_info(fine)
        mask_type, mask_rows, mask_cols = mat_info(fine_mask)
        gdb.write(
            f"UP_ENTRY {state['up']} coarse={coarse_cols}x{coarse_rows}/type{coarse_type} "
            f"fine={fine_cols}x{fine_rows}/type{fine_type} "
            f"mask={mask_cols}x{mask_rows}/type{mask_type}\n"
        )
        return False


WrapperEntry(f"*0x{pie_base + 0x22C2D0:x}")
DownEntry(f"*0x{pie_base + 0x229AB0:x}")
UpEntry(f"*0x{pie_base + 0x229560:x}")
end

continue
