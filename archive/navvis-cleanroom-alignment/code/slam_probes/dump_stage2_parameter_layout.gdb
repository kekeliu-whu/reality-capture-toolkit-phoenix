set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
import os
import struct


EXECUTABLE = "/opt/NavVis/slam/lib/surveyor_ros/compute_trajectories"
# Public ceres::Solve wrapper in this frozen compute_trajectories Build ID.
CERES_SOLVE_OFFSET = 0x324E40


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def executable_base():
    pid = gdb.selected_inferior().pid
    with open(f"/proc/{pid}/maps", "r", encoding="ascii") as stream:
        for line in stream:
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or fields[5].strip() != EXECUTABLE:
                continue
            start, _ = (int(value, 16) for value in fields[0].split("-"))
            file_offset = int(fields[2], 16)
            return start - file_offset
    raise gdb.GdbError("compute_trajectories executable mapping was not found")


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        address = executable_base() + CERES_SOLVE_OFFSET
        super().__init__(f"*{address:#x}", internal=True)

    def stop(self):
        options = int(gdb.parse_and_eval("$rdi"))
        problem = int(gdb.parse_and_eval("$rsi"))
        option_bytes = read(options, 488)
        gdb.write("SOLVER_OPTIONS_HEX " + option_bytes.hex() + "\n")
        fields = (
            ("max_num_iterations", 104, "i"),
            ("num_threads", 120, "i"),
            ("function_tolerance", 184, "d"),
            ("gradient_tolerance", 192, "d"),
            ("parameter_tolerance", 200, "d"),
            ("linear_solver_type", 208, "i"),
            ("preconditioner_type", 212, "i"),
            ("min_linear_solver_iterations", 344, "i"),
            ("max_linear_solver_iterations", 348, "i"),
            ("eta", 352, "d"),
        )
        for name, offset, code in fields:
            value = struct.unpack_from("<" + code, option_bytes, offset)[0]
            gdb.write(f"SOLVER_OPTION name={name} value={value}\n")

        implementation = u64(problem)
        program = u64(implementation + 160)
        begin = u64(program)
        end = u64(program + 8)
        if end < begin or (end - begin) % 8:
            raise gdb.GdbError("invalid Ceres parameter-block vector")
        pointers = struct.unpack(f"<{(end - begin) // 8}Q", read(begin, end - begin))
        sizes = []
        for pointer in pointers:
            sizes.append(struct.unpack("<i", read(pointer + 8, 4))[0])
        gdb.write(f"PARAMETER_VECTOR count={len(pointers)}\n")

        run_start = 0
        for index in range(1, len(sizes) + 1):
            if index < len(sizes) and sizes[index] == sizes[run_start]:
                continue
            gdb.write(
                f"PARAMETER_RUN begin={run_start} end={index} "
                f"size={sizes[run_start]} count={index - run_start}\n"
            )
            run_start = index

        selected = set(range(min(20, len(pointers))))
        selected.update(range(max(0, len(pointers) - 24), len(pointers)))
        for index in range(1, len(pointers)):
            if sizes[index] != sizes[index - 1]:
                selected.update((index - 1, index))
        for index in sorted(selected):
            pointer = pointers[index]
            storage = u64(pointer)
            size = sizes[index]
            values = struct.unpack(f"<{size}d", read(storage, size * 8))
            gdb.write(
                f"PARAMETER index={index} size={size} storage={storage:#x} values="
                + " ".join(format(value, ".17g") for value in values)
                + "\n"
            )
        gdb.execute("quit")
        return False


SolveEntry()
end

continue
