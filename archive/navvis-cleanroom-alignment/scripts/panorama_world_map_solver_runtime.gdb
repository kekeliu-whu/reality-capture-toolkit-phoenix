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
        mapping_start = int(fields[0].split("-", 1)[0], 16)
        file_offset = int(fields[2], 16)
        candidate = mapping_start - file_offset
        pie_base = candidate if pie_base is None else min(pie_base, candidate)

if pie_base is None:
    raise gdb.GdbError("could not locate renderer PIE base")

dump_directory = os.environ.get("NAVVIS_WORLD_MAP_SOLVER_RUNTIME_DIR")
if not dump_directory:
    raise gdb.GdbError("NAVVIS_WORLD_MAP_SOLVER_RUNTIME_DIR is required")
os.makedirs(dump_directory, exist_ok=True)

state = {"level": 0, "outer": 0, "inner": 0, "count": 0}


def read_u64(address):
    return struct.unpack("<Q", inferior.read_memory(address, 8))[0]


def dump_vector(stack, offset, label):
    data = read_u64(stack + offset)
    count = read_u64(stack + offset + 8)
    if data == 0 or count == 0:
        return
    path = os.path.join(
        dump_directory,
        f"level{state['level']}_outer{state['outer']}_inner{state['inner']}_{label}.raw",
    )
    with open(path, "wb") as output:
        output.write(inferior.read_memory(data, count * 8))


def dump_context(stack, stage):
    for offset, label in (
        (0x50, "hessian_direction"),
        (0x60, "residual"),
        (0x70, "direction"),
        (0x80, "delta"),
        (0x98, "preconditioned"),
        (0xA8, "inverse_diagonal"),
    ):
        dump_vector(stack, offset, f"{stage}_{label}")


class SolverEntry(gdb.Breakpoint):
    def stop(self):
        state["level"] += 1
        state["outer"] = 0
        state["inner"] = 0
        measured = int(gdb.selected_frame().read_register("rsi"))
        rows = struct.unpack("<i", inferior.read_memory(measured + 8, 4))[0]
        columns = struct.unpack("<i", inferior.read_memory(measured + 12, 4))[0]
        state["count"] = rows * columns
        gdb.write(f"solver-runtime level={state['level']} size={columns}x{rows}\n")
        return False


class GradientReady(gdb.Breakpoint):
    def stop(self):
        if state["outer"] == 0:
            stack = int(gdb.selected_frame().read_register("rsp"))
            dump_context(stack, "gradient_ready")
        return False


class FirstStepReady(gdb.Breakpoint):
    def stop(self):
        if state["outer"] == 0 and state["inner"] == 0:
            stack = int(gdb.selected_frame().read_register("rsp"))
            dump_context(stack, "first_hessian_ready")
            gdb.write(
                f"solver-runtime captured level{state['level']} initial state "
                "and first Hessian application\n"
            )
            return state["level"] == 4
        return False


class InnerTail(gdb.Breakpoint):
    def stop(self):
        state["inner"] += 1
        return False


class OuterObjective(gdb.Breakpoint):
    def stop(self):
        state["outer"] += 1
        state["inner"] = 0
        return False


SolverEntry(f"*0x{pie_base + 0x211ee0:x}")
GradientReady(f"*0x{pie_base + 0x21202f:x}")
FirstStepReady(f"*0x{pie_base + 0x212aea:x}")
InnerTail(f"*0x{pie_base + 0x213f71:x}")
OuterObjective(f"*0x{pie_base + 0x2144d9:x}")
end

continue
