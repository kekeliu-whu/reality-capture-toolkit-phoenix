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
        values = {
            "termination": struct.unpack_from("<i", raw, 4)[0],
            "initial_cost": struct.unpack_from("<d", raw, 40)[0],
            "final_cost": struct.unpack_from("<d", raw, 48)[0],
            "successful_steps": struct.unpack_from("<i", raw, 88)[0],
            "unsuccessful_steps": struct.unpack_from("<i", raw, 92)[0],
        }
        (out_dir / "clean_solver_summary.json").write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        iteration_begin, iteration_end = struct.unpack_from("<QQ", raw, 64)
        stride = 120
        if iteration_end < iteration_begin or (iteration_end - iteration_begin) % stride:
            raise gdb.GdbError(
                "unexpected IterationSummary vector: begin=%#x end=%#x"
                % (iteration_begin, iteration_end)
            )
        iteration_raw = inferior.read_memory(
            iteration_begin, iteration_end - iteration_begin
        ).tobytes()
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
        options_address = int(gdb.parse_and_eval("$rdi"))
        summary_address = int(gdb.parse_and_eval("$rdx"))
        # Ceres 2.0, 2.1, and 2.2 all place Options::num_threads at +120.
        # Force the solver itself to one thread even for older workers whose
        # CLI predates --exposure-solver-threads.
        original_threads = struct.unpack(
            "<i", inferior.read_memory(options_address + 120, 4).tobytes()
        )[0]
        inferior.write_memory(options_address + 120, struct.pack("<i", 1))
        if max_iterations_override:
            inferior.write_memory(
                options_address + 104,
                struct.pack("<i", int(max_iterations_override)),
            )
        (out_dir / "solver_thread_override.json").write_text(
            json.dumps(
                {
                    "options_address": options_address,
                    "num_threads_offset": 120,
                    "original": original_threads,
                    "effective": 1,
                    "max_iterations_override": (
                        int(max_iterations_override) if max_iterations_override else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        SolveReturnBreakpoint(gdb.newest_frame(), summary_address)
        self.enabled = False
        return False


class ExposureSolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        inferior = gdb.selected_inferior()
        # The std::vector<GammaModel> return value uses the SysV hidden sret
        # pointer in RDI, so the first explicit argument (ExposureProblem) is
        # in RSI at the machine-code entry.
        argument_address = int(gdb.parse_and_eval("$rsi"))

        # Locate the Scene vector structurally before writing.  Depending on
        # which local clone GDB resolves, RSI can point one std::vector before
        # the ExposureProblem base.  Scene is uniquely identifiable here as
        # 136 contiguous 16-byte ranges with view ids 0..135 and low < high.
        stride = 16
        candidates = []
        for relative_offset in range(-48, 145, 24):
            vector_address = argument_address + relative_offset
            try:
                vector_raw = inferior.read_memory(vector_address, 24).tobytes()
                candidate_begin, candidate_end, candidate_capacity = struct.unpack(
                    "<QQQ", vector_raw
                )
                if (
                    candidate_end < candidate_begin
                    or candidate_capacity < candidate_end
                    or (candidate_end - candidate_begin) != 136 * stride
                ):
                    continue
                valid = True
                views = []
                for index in range(136):
                    raw = inferior.read_memory(
                        candidate_begin + index * stride, stride
                    ).tobytes()
                    view = struct.unpack_from("<i", raw, 0)[0]
                    low, high = struct.unpack_from("<BB", raw, 4)
                    if not (0 <= view < 136 and low < high):
                        valid = False
                        break
                    views.append(view)
                if valid and sorted(views) == list(range(136)):
                    candidates.append(
                        (
                            relative_offset,
                            vector_address,
                            candidate_begin,
                            candidate_end,
                            candidate_capacity,
                        )
                    )
            except gdb.MemoryError:
                continue
        if len(candidates) != 1:
            raise gdb.GdbError(
                "expected exactly one structural Scene vector, found %d" % len(candidates)
            )
        relative_offset, vector_address, begin, end, capacity = candidates[0]
        count = (end - begin) // stride

        rows = []
        changed = 0
        maximum_delta = 0.0
        for index in range(count):
            item_address = begin + index * stride
            raw = inferior.read_memory(item_address, stride).tobytes()
            view = struct.unpack_from("<i", raw, 0)[0]
            low, high, median = struct.unpack_from("<BBB", raw, 4)
            original = struct.unpack_from("<d", raw, 8)[0]
            rounded = float(struct.unpack("<f", struct.pack("<f", original))[0])
            delta = rounded - original
            maximum_delta = max(maximum_delta, abs(delta))
            if rounded != original:
                inferior.write_memory(item_address + 8, struct.pack("<d", rounded))
                changed += 1
            rows.append(
                {
                    "index": index,
                    "view": view,
                    "low": low,
                    "high": high,
                    "median": median,
                    "weight_before": original,
                    "weight_after": rounded,
                    "delta": delta,
                }
            )

        capture = {
            "argument_address": argument_address,
            "scene_vector_address": vector_address,
            "scene_vector_relative_offset": relative_offset,
            "scene_begin": begin,
            "scene_end": end,
            "scene_capacity": capacity,
            "scene_count": count,
            "scene_stride": stride,
            "changed_count": changed,
            "maximum_abs_delta": maximum_delta,
            "rows": rows,
        }
        (out_dir / "scene_weight_runtime_patch.json").write_text(
            json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.enabled = False
        return False


ExposureSolveBreakpoint("solveExposureProblem", internal=True)
CeresSolveBreakpoint(
    "ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
    internal=True,
)
end

run
quit
