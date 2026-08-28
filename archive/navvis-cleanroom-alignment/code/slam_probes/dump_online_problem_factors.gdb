set pagination off
set confirm off
set logging enabled off
set print thread-events off
starti

python
import gdb
import os
import pathlib
import struct


IMAGE_BASE = 0x555555554000
CERES_SOLVE = IMAGE_BASE + 0xEDD070
TARGET_CERES_CALL = int(os.environ.get("NAVVIS_PROBE_CERES_CALL", "17"))
OUTPUT = pathlib.Path(os.environ["NAVVIS_PROBE_FACTORS"])


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def i32(address):
    return struct.unpack("<i", read(address, 4))[0]


def parameter_sizes(cost):
    begin = u64(cost + 8)
    end = u64(cost + 16)
    if end < begin or (end - begin) % 4:
        return ()
    count = (end - begin) // 4
    if count > 16:
        return ()
    return struct.unpack(f"<{count}i", read(begin, end - begin))


def dump_problem(problem):
    implementation = u64(problem)
    program = u64(implementation + 160)
    residual_begin = u64(program + 24)
    residual_end = u64(program + 32)
    if residual_end < residual_begin or (residual_end - residual_begin) % 8:
        raise gdb.GdbError("invalid Ceres Program residual vector")
    residuals = struct.unpack(
        f"<{(residual_end - residual_begin) // 8}Q",
        read(residual_begin, residual_end - residual_begin),
    )
    counts = {}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="ascii") as stream:
        for order, residual in enumerate(residuals):
            cost = u64(residual)
            loss = u64(residual + 8)
            sizes = parameter_sizes(cost)
            dimension = i32(cost + 32)
            signature = (dimension, sizes)
            counts[signature] = counts.get(signature, 0) + 1
            if signature != (6, (4, 3, 4, 3)):
                continue
            functor = u64(cost + 40)
            values = struct.unpack("<9d", read(functor, 72))
            stream.write(
                "GRAPH_FACTOR "
                + str(order)
                + " "
                + str(int(loss != 0))
                + " "
                + " ".join(format(value, ".17g") for value in values)
                + "\n"
            )
    gdb.write(
        f"ONLINE_PROBLEM_FACTORS residuals={len(residuals)} "
        f"signatures={counts} output={OUTPUT}\n"
    )


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{CERES_SOLVE:#x}", internal=True)
        self.count = 0

    def stop(self):
        self.count += 1
        if self.count == TARGET_CERES_CALL:
            dump_problem(int(gdb.parse_and_eval("$rsi")))
            gdb.execute("quit")
        return False


SolveEntry()
end

continue
