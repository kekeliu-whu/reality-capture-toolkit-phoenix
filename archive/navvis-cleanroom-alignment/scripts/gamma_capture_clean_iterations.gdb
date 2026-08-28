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
max_iterations_override = os.environ.get("GAMMA_MAX_ITERATIONS", "")


class SolveReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, summary_address):
        super().__init__(frame, internal=True)
        self.summary_address = summary_address

    def stop(self):
        inferior = gdb.selected_inferior()
        raw = inferior.read_memory(self.summary_address, 512).tobytes()
        (out_dir / "solver_summary_512.bin").write_bytes(raw)
        summary = {
            "termination": struct.unpack_from("<i", raw, 4)[0],
            "initial_cost": struct.unpack_from("<d", raw, 40)[0],
            "final_cost": struct.unpack_from("<d", raw, 48)[0],
            "successful_steps": struct.unpack_from("<i", raw, 88)[0],
            "unsuccessful_steps": struct.unpack_from("<i", raw, 92)[0],
        }
        (out_dir / "clean_solver_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        begin, end = struct.unpack_from("<QQ", raw, 64)
        stride = 120
        if end < begin or (end - begin) % stride:
            raise gdb.GdbError(
                "unexpected IterationSummary vector: begin=%#x end=%#x" % (begin, end)
            )
        iteration_raw = inferior.read_memory(begin, end - begin).tobytes()
        (out_dir / "iteration_summaries.bin").write_bytes(iteration_raw)
        iterations = []
        for offset in range(0, len(iteration_raw), stride):
            iterations.append(
                {
                    "iteration": struct.unpack_from("<i", iteration_raw, offset)[0],
                    "step_is_valid": bool(iteration_raw[offset + 4]),
                    "step_is_nonmonotonic": bool(iteration_raw[offset + 5]),
                    "step_is_successful": bool(iteration_raw[offset + 6]),
                    "cost": struct.unpack_from("<d", iteration_raw, offset + 8)[0],
                    "cost_change": struct.unpack_from("<d", iteration_raw, offset + 16)[0],
                    "gradient_max_norm": struct.unpack_from("<d", iteration_raw, offset + 24)[0],
                    "gradient_norm": struct.unpack_from("<d", iteration_raw, offset + 32)[0],
                    "step_norm": struct.unpack_from("<d", iteration_raw, offset + 40)[0],
                    "relative_decrease": struct.unpack_from("<d", iteration_raw, offset + 48)[0],
                    "trust_region_radius": struct.unpack_from("<d", iteration_raw, offset + 56)[0],
                    "eta": struct.unpack_from("<d", iteration_raw, offset + 64)[0],
                    "linear_solver_iterations": struct.unpack_from("<i", iteration_raw, offset + 92)[0],
                }
            )
        (out_dir / "iteration_summaries.json").write_text(
            json.dumps(iterations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return False


class CeresSolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        inferior = gdb.selected_inferior()
        options = int(gdb.parse_and_eval("$rdi"))
        summary = int(gdb.parse_and_eval("$rdx"))
        original_threads = struct.unpack(
            "<i", inferior.read_memory(options + 120, 4).tobytes()
        )[0]
        original_max_iterations = struct.unpack(
            "<i", inferior.read_memory(options + 104, 4).tobytes()
        )[0]
        inferior.write_memory(options + 120, struct.pack("<i", 1))
        if max_iterations_override:
            inferior.write_memory(
                options + 104, struct.pack("<i", int(max_iterations_override))
            )
        (out_dir / "solver_overrides.json").write_text(
            json.dumps(
                {
                    "num_threads_original": original_threads,
                    "num_threads_effective": 1,
                    "max_iterations_original": original_max_iterations,
                    "max_iterations_effective": (
                        int(max_iterations_override)
                        if max_iterations_override
                        else original_max_iterations
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        SolveReturnBreakpoint(gdb.newest_frame(), summary)
        self.enabled = False
        return False


CeresSolveBreakpoint(
    "ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
    internal=True,
)
end

run
quit
