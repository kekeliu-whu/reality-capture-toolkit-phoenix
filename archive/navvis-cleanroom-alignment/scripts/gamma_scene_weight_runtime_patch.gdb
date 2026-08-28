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


class SolveReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, summary_address):
        super().__init__(frame, internal=True)
        self.summary_address = summary_address

    def stop(self):
        raw = gdb.selected_inferior().read_memory(self.summary_address, 512).tobytes()
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
        return False


class CeresSolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        summary_address = int(gdb.parse_and_eval("$rdx"))
        SolveReturnBreakpoint(gdb.newest_frame(), summary_address)
        self.enabled = False
        return False


class ExposureSolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        inferior = gdb.selected_inferior()
        # The std::vector<GammaModel> return value uses the SysV hidden sret
        # pointer in RDI, so the first explicit argument (ExposureProblem) is
        # in RSI at the machine-code entry.
        problem_address = int(gdb.parse_and_eval("$rsi"))

        # ExposureProblem is four std::vectors.  scene_ranges is the third
        # vector, at +48.  The current clean ExposureSceneRange is 16 bytes:
        # int view; uint8 low/high/median; padding; double normalized_weight.
        vector_raw = inferior.read_memory(problem_address + 48, 24).tobytes()
        begin, end, capacity = struct.unpack("<QQQ", vector_raw)
        stride = 16
        if end < begin or (end - begin) % stride != 0:
            raise gdb.GdbError(
                "unexpected ExposureSceneRange vector: begin=%#x end=%#x" % (begin, end)
            )
        count = (end - begin) // stride
        if count == 0 or capacity < end:
            raise gdb.GdbError(
                "invalid ExposureSceneRange vector: count=%d capacity=%#x" % (count, capacity)
            )

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
            "problem_address": problem_address,
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
