set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
import struct


EXECUTABLE = "/opt/NavVis/slam/lib/surveyor_ros/compute_trajectories"
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
            return start - int(fields[2], 16)
    raise gdb.GdbError("compute_trajectories executable mapping was not found")


class SolveFinish(gdb.FinishBreakpoint):
    def __init__(self, frame, blocks):
        super().__init__(frame, internal=True)
        self.blocks = blocks

    def stop(self):
        for index, size, storage in self.blocks:
            values = struct.unpack(f"<{size}d", read(storage, size * 8))
            gdb.write(
                f"FINAL_PARAMETER index={index} size={size} values="
                + " ".join(format(value, ".17g") for value in values)
                + "\n"
            )
        gdb.execute("quit")
        return False


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{executable_base() + CERES_SOLVE_OFFSET:#x}", internal=True)

    def stop(self):
        problem = int(gdb.parse_and_eval("$rsi"))
        implementation = u64(problem)
        program = u64(implementation + 160)
        begin = u64(program)
        end = u64(program + 8)
        pointers = struct.unpack(f"<{(end - begin) // 8}Q", read(begin, end - begin))
        blocks = []
        for index in range(len(pointers) - 9, len(pointers)):
            parameter = pointers[index]
            storage = u64(parameter)
            size = struct.unpack("<i", read(parameter + 8, 4))[0]
            blocks.append((index, size, storage))
        SolveFinish(gdb.newest_frame(), blocks)
        self.enabled = False
        return False


SolveEntry()
end

continue
