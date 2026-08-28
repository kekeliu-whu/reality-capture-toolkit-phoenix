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
output_directory = os.environ.get("NAVVIS_WORLD_MAP_STAGE_DUMP_DIR")
if not output_directory:
    raise gdb.GdbError("NAVVIS_WORLD_MAP_STAGE_DUMP_DIR is required")
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


def read_u64(address):
    return struct.unpack("<Q", inferior.read_memory(address, 8))[0]


def mat_metadata(mat_address):
    raw = bytes(inferior.read_memory(mat_address, 96))
    flags, dims, rows, cols = struct.unpack_from("<iiii", raw, 0)
    data = struct.unpack_from("<Q", raw, 16)[0]
    step_pointer = struct.unpack_from("<Q", raw, 72)[0]
    row_step = read_u64(step_pointer)
    return flags & 0xFFF, dims, rows, cols, data, row_step


class WorldMapWorkerEntry(gdb.Breakpoint):
    def __init__(self, address):
        super().__init__(f"*0x{address:x}")
        self.serial = 0
        self.seen = set()

    def stop(self):
        work_range = int(gdb.parse_and_eval("$rdi"))
        row_begin, row_end = struct.unpack(
            "<ii", inferior.read_memory(work_range, 8)
        )
        closure = read_u64(work_range + 8)
        matrix_array_pointer = read_u64(closure + 0x18)
        matrix_array = read_u64(matrix_array_pointer)
        map_index_pointer = read_u64(closure + 0x20)
        map_index = read_u64(map_index_pointer)
        mat_address = matrix_array + map_index * 96
        mat_type, dims, rows, cols, data, row_step = mat_metadata(mat_address)
        key = (data, rows, cols, mat_type)
        if key in self.seen:
            return False
        self.seen.add(key)

        serial = self.serial
        self.serial += 1
        gdb.write(
            f"WORLD_MAP_STAGE serial={serial} rows={row_begin}:{row_end} "
            f"index={map_index} type={mat_type} dims={dims} "
            f"shape={cols}x{rows} step={row_step}\n"
        )
        if mat_type != 21 or dims != 2 or row_step < cols * 12:
            return False

        raw = bytes(inferior.read_memory(data, rows * row_step))
        payload = b"".join(
            raw[row * row_step:row * row_step + cols * 12] for row in range(rows)
        )
        path = os.path.join(
            output_directory, f"world_map_stage_{serial:02d}_{cols}x{rows}.f32c3"
        )
        with open(path, "wb") as output:
            output.write(payload)
        gdb.write(
            f"WORLD_MAP_STAGE_DUMP serial={serial} bytes={len(payload)} "
            f"sha256={hashlib.sha256(payload).hexdigest()} {path}\n"
        )
        return False


WorldMapWorkerEntry(pie_base + 0x1ACB20)
end

continue
