set pagination off
set confirm off
set logging enabled off
set print thread-events off
starti

python
import gdb
import pathlib
import struct


IMAGE_BASE = 0x555555554000
CERES_SOLVE = IMAGE_BASE + 0xEDD070


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


def problem_signature(problem):
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
    signatures = {}
    graph_with_loss = 0
    graph_without_loss = 0
    graph_tail = None
    for residual in residuals:
        cost = u64(residual)
        loss = u64(residual + 8)
        signature = (i32(cost + 32), parameter_sizes(cost))
        signatures[signature] = signatures.get(signature, 0) + 1
        if signature != (6, (4, 3, 4, 3)):
            continue
        if loss:
            graph_with_loss += 1
        else:
            graph_without_loss += 1
        functor = u64(cost + 40)
        # GraphResidual stores translation, one ABI-alignment double,
        # quaternion, then both scalar weights.
        graph_tail = struct.unpack("<10d", read(functor, 80))
    return (
        len(residuals),
        signatures,
        graph_without_loss,
        graph_with_loss,
        graph_tail,
    )


class SolveReturn(gdb.FinishBreakpoint):
    def __init__(self, call, summary):
        super().__init__(gdb.newest_frame(), internal=True)
        self.call = call
        self.summary = summary

    def stop(self):
        summary = read(self.summary, 104)
        gdb.write(
            "CERES_RETURN "
            f"call={self.call} "
            f"termination={struct.unpack_from('<i', summary, 4)[0]} "
            f"initial_cost={struct.unpack_from('<d', summary, 40)[0]:.17g} "
            f"final_cost={struct.unpack_from('<d', summary, 48)[0]:.17g} "
            f"successful_steps={struct.unpack_from('<i', summary, 88)[0]} "
            f"unsuccessful_steps={struct.unpack_from('<i', summary, 92)[0]}\n"
        )
        return False


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{CERES_SOLVE:#x}", internal=True)
        self.count = 0

    def stop(self):
        self.count += 1
        options_address = int(gdb.parse_and_eval("$rdi"))
        problem_address = int(gdb.parse_and_eval("$rsi"))
        summary_address = int(gdb.parse_and_eval("$rdx"))
        options = read(options_address, 216)
        residual_count, signatures, graph_plain, graph_loss, tail = problem_signature(
            problem_address
        )
        tail_text = "none" if tail is None else ",".join(
            format(value, ".17g") for value in tail
        )
        gdb.write(
            "CERES_ENTRY "
            f"call={self.count} "
            f"max_iterations={struct.unpack_from('<i', options, 104)[0]} "
            f"threads={struct.unpack_from('<i', options, 120)[0]} "
            f"residuals={residual_count} "
            f"graph_plain={graph_plain} graph_loss={graph_loss} "
            f"signatures={signatures} graph_tail={tail_text}\n"
        )
        SolveReturn(self.count, summary_address)
        return False


SolveEntry()
end

continue
