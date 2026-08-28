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


out_dir = pathlib.Path(os.environ["GAMMA_CERES_FINGERPRINT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
ceres_version = os.environ["GAMMA_CERES_FINGERPRINT_VERSION"]
patch_scene_weights = os.environ.get("GAMMA_CERES_FINGERPRINT_PATCH_SCENE", "1") == "1"


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


class SolveReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, summary_address):
        super().__init__(frame, internal=True)
        self.summary_address = summary_address

    def stop(self):
        raw = read(self.summary_address, 512)
        (out_dir / "solver_summary_512.bin").write_bytes(raw)
        iterations_begin, iterations_end, iterations_capacity = struct.unpack_from(
            "<QQQ", raw, 64
        )
        iteration_stride = 120
        if (iterations_end < iterations_begin or
                (iterations_end - iterations_begin) % iteration_stride != 0 or
                iterations_capacity < iterations_end):
            raise gdb.GdbError(
                "unexpected iteration vector begin=%#x end=%#x capacity=%#x"
                % (iterations_begin, iterations_end, iterations_capacity)
            )
        iterations_raw = read(iterations_begin, iterations_end - iterations_begin)
        (out_dir / "iterations.bin").write_bytes(iterations_raw)
        iterations = []
        for offset in range(0, len(iterations_raw), iteration_stride):
            item = iterations_raw[offset : offset + iteration_stride]
            iterations.append({
                "iteration": struct.unpack_from("<i", item, 0)[0],
                "step_is_valid": bool(item[4]),
                "step_is_nonmonotonic": bool(item[5]),
                "step_is_successful": bool(item[6]),
                "cost": struct.unpack_from("<d", item, 8)[0],
                "cost_change": struct.unpack_from("<d", item, 16)[0],
                "gradient_max_norm": struct.unpack_from("<d", item, 24)[0],
                "gradient_norm": struct.unpack_from("<d", item, 32)[0],
                "step_norm": struct.unpack_from("<d", item, 40)[0],
                "relative_decrease": struct.unpack_from("<d", item, 48)[0],
                "trust_region_radius": struct.unpack_from("<d", item, 56)[0],
                "eta": struct.unpack_from("<d", item, 64)[0],
                "linear_solver_iterations": struct.unpack_from("<i", item, 92)[0],
            })
        (out_dir / "iterations.json").write_text(
            json.dumps(iterations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        values = {
            "ceres_version": ceres_version,
            "termination": struct.unpack_from("<i", raw, 4)[0],
            "initial_cost": struct.unpack_from("<d", raw, 40)[0],
            "final_cost": struct.unpack_from("<d", raw, 48)[0],
            "successful_steps": struct.unpack_from("<i", raw, 88)[0],
            "unsuccessful_steps": struct.unpack_from("<i", raw, 92)[0],
        }
        (out_dir / "clean_solver_summary.json").write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return False


class CeresSolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        options_address = int(gdb.parse_and_eval("$rdi"))
        summary_address = int(gdb.parse_and_eval("$rdx"))
        options_size = 504 if ceres_version == "2.2.0" else 488
        progress_offset = 388 if ceres_version == "2.2.0" else 368
        logging_offset = 384 if ceres_version == "2.2.0" else 364
        raw_before = read(options_address, options_size)
        (out_dir / "solver_options_before.bin").write_bytes(raw_before)

        # Logging does not enter the objective or linear solve.  It exposes the
        # initial gradient and every accepted/rejected LM step for fingerprinting.
        gdb.selected_inferior().write_memory(options_address + logging_offset,
                                             struct.pack("<i", 1))
        gdb.selected_inferior().write_memory(options_address + progress_offset, b"\x01")
        (out_dir / "solver_options_effective.bin").write_bytes(
            read(options_address, options_size)
        )
        SolveReturnBreakpoint(gdb.newest_frame(), summary_address)
        self.enabled = False
        return False


class ExposureSolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        inferior = gdb.selected_inferior()
        # SysV hidden sret in RDI; ExposureProblem is the first explicit
        # argument in RSI.  Its third std::vector, scene_ranges, starts at +48.
        problem_address = int(gdb.parse_and_eval("$rsi"))
        begin, end, capacity = struct.unpack("<QQQ", read(problem_address + 48, 24))
        stride = 16
        if end < begin or (end - begin) % stride != 0 or capacity < end:
            raise gdb.GdbError(
                "unexpected scene vector begin=%#x end=%#x capacity=%#x"
                % (begin, end, capacity)
            )

        rows = []
        for index, address in enumerate(range(begin, end, stride)):
            raw = read(address, stride)
            view = struct.unpack_from("<i", raw, 0)[0]
            low, high, median = struct.unpack_from("<BBB", raw, 4)
            original = struct.unpack_from("<d", raw, 8)[0]
            rounded = float(struct.unpack("<f", struct.pack("<f", original))[0])
            if rounded != original:
                inferior.write_memory(address + 8, struct.pack("<d", rounded))
            rows.append({
                "index": index,
                "view": view,
                "low": low,
                "high": high,
                "median": median,
                "weight_before": original,
                "weight_after": rounded,
            })

        (out_dir / "scene_weight_runtime_patch.json").write_text(
            json.dumps({
                "problem_address": problem_address,
                "scene_count": len(rows),
                "scene_stride": stride,
                "rows": rows,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return False


if patch_scene_weights:
    ExposureSolveBreakpoint("solveExposureProblem", internal=True)
CeresSolveBreakpoint(
    "ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
    internal=True,
)
end

run
quit
