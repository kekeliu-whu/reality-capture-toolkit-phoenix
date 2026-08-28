set pagination off
set confirm off
set breakpoint pending on
set disable-randomization on
set print thread-events off

python
import json
import os
import pathlib
import struct

import gdb


out_dir = pathlib.Path(os.environ["GAMMA_AUTO_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
dump_iteration = int(os.environ.get("GAMMA_DUMP_ITERATION", "1"))
max_iterations = int(os.environ.get("GAMMA_MAX_ITERATIONS", str(dump_iteration)))
initial_radius = float(os.environ.get("GAMMA_INITIAL_RADIUS", "10000"))

VECTOR_OFFSET = 376
STRING_OFFSET = 400
DUMP_FORMAT_OFFSET = 432


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def write(address, data):
    gdb.selected_inferior().write_memory(address, data)


class SolveFinish(gdb.FinishBreakpoint):
    def __init__(self, frame, options, summary, saved_vector, saved_string, saved_format):
        super().__init__(frame, internal=True)
        self.options = options
        self.summary = summary
        self.saved_vector = saved_vector
        self.saved_string = saved_string
        self.saved_format = saved_format

    def stop(self):
        write(self.options + VECTOR_OFFSET, self.saved_vector)
        write(self.options + STRING_OFFSET, self.saved_string)
        write(self.options + DUMP_FORMAT_OFFSET, self.saved_format)

        raw = read(self.summary, 512)
        (out_dir / "solver_summary_512.bin").write_bytes(raw)
        summary = {
            "termination": struct.unpack_from("<i", raw, 4)[0],
            "initial_cost": struct.unpack_from("<d", raw, 40)[0],
            "final_cost": struct.unpack_from("<d", raw, 48)[0],
            "successful_steps": struct.unpack_from("<i", raw, 88)[0],
            "unsuccessful_steps": struct.unpack_from("<i", raw, 92)[0],
        }
        (out_dir / "solver_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return False


class SolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        options = int(gdb.parse_and_eval("$rdi"))
        summary = int(gdb.parse_and_eval("$rdx"))
        saved_vector = read(options + VECTOR_OFFSET, 24)
        saved_string = read(options + STRING_OFFSET, 32)
        saved_format = read(options + DUMP_FORMAT_OFFSET, 4)

        # The source Options object already owns these empty containers.  Point
        # the vector temporarily at max_num_iterations, whose value is also the
        # requested dump iteration, and restore all ownership fields before the
        # caller can destroy Options.
        write(options + 104, struct.pack("<i", max_iterations))
        write(options + 120, struct.pack("<i", 1))
        write(options + 128, struct.pack("<d", initial_radius))
        iteration_storage = options + 104
        write(
            options + VECTOR_OFFSET,
            struct.pack("<QQQ", iteration_storage, iteration_storage + 4,
                        iteration_storage + 4),
        )

        # Run the inferior with cwd=out_dir and use the libstdc++ SSO layout for
        # ".".  The copied Minimizer::Options receives an owning copy.
        string_object = options + STRING_OFFSET
        write(string_object, struct.pack("<Q", string_object + 16))
        write(string_object + 8, struct.pack("<Q", 1))
        write(string_object + 16, b".\0" + b"\0" * 14)
        write(options + DUMP_FORMAT_OFFSET, struct.pack("<i", 1))

        metadata = {
            "dump_iteration": dump_iteration,
            "max_iterations": max_iterations,
            "num_threads": 1,
            "initial_trust_region_radius": initial_radius,
            "options_address": hex(options),
            "summary_address": hex(summary),
        }
        (out_dir / "dump_patch.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        SolveFinish(
            gdb.newest_frame(), options, summary, saved_vector, saved_string, saved_format
        )
        self.enabled = False
        return False


SolveBreakpoint(
    "ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
    internal=True,
)
end

run
quit
