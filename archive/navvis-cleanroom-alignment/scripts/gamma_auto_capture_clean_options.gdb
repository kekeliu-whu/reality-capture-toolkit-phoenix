set pagination off
set confirm off
set breakpoint pending on
set disable-randomization on
set print thread-events off

python
import os
import pathlib
import struct
import gdb

out_dir = pathlib.Path(os.environ["GAMMA_AUTO_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)


class OptionsBreakpoint(gdb.Breakpoint):
    def stop(self):
        options = int(gdb.parse_and_eval("$rdi"))
        raw = gdb.selected_inferior().read_memory(options, 504).tobytes()
        (out_dir / "solver_options_504.bin").write_bytes(raw)
        fields = [
            ("minimizer_type", 0, "i"),
            ("trust_region_strategy_type", 88, "i"),
            ("use_nonmonotonic_steps", 96, "B"),
            ("max_consecutive_nonmonotonic_steps", 100, "i"),
            ("max_num_iterations", 104, "i"),
            ("max_solver_time_in_seconds", 112, "d"),
            ("num_threads", 120, "i"),
            ("initial_trust_region_radius", 128, "d"),
            ("max_trust_region_radius", 136, "d"),
            ("min_trust_region_radius", 144, "d"),
            ("min_relative_decrease", 152, "d"),
            ("min_lm_diagonal", 160, "d"),
            ("max_lm_diagonal", 168, "d"),
            ("max_num_consecutive_invalid_steps", 176, "i"),
            ("function_tolerance", 184, "d"),
            ("gradient_tolerance", 192, "d"),
            ("parameter_tolerance", 200, "d"),
            ("linear_solver_type", 208, "i"),
            ("preconditioner_type", 212, "i"),
            ("min_linear_solver_iterations", 320, "i"),
            ("max_linear_solver_iterations", 324, "i"),
            ("eta", 344, "d"),
            ("jacobi_scaling", 352, "B"),
            ("logging_type", 384, "i"),
            ("minimizer_progress_to_stdout", 388, "B"),
        ]
        with (out_dir / "clean_solver_options.txt").open("w", encoding="utf-8") as output:
            for name, offset, code in fields:
                value = struct.unpack_from("<" + code, raw, offset)[0]
                output.write("OPTION %s offset=%d value=%s\n" % (name, offset, value))
        self.enabled = False
        return True


OptionsBreakpoint("ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
                  internal=True)
end

run
quit
