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


class SolveReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, summary):
        super().__init__(frame, internal=True)
        self.summary = summary

    def stop(self):
        raw = gdb.selected_inferior().read_memory(self.summary, 512).tobytes()
        (out_dir / "solver_summary_512.bin").write_bytes(raw)
        values = {
            "termination": struct.unpack_from("<i", raw, 4)[0],
            "initial_cost": struct.unpack_from("<d", raw, 40)[0],
            "final_cost": struct.unpack_from("<d", raw, 48)[0],
            "successful_steps": struct.unpack_from("<i", raw, 88)[0],
            "unsuccessful_steps": struct.unpack_from("<i", raw, 92)[0],
        }
        with (out_dir / "clean_solver_summary.txt").open("w", encoding="utf-8") as output:
            output.write(
                "SUMMARY termination={termination} initial_cost={initial_cost:.17g} "
                "final_cost={final_cost:.17g} successful_steps={successful_steps} "
                "unsuccessful_steps={unsuccessful_steps}\n".format(**values)
            )
        return True


class SolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        summary = int(gdb.parse_and_eval("$rdx"))
        SolveReturnBreakpoint(gdb.newest_frame(), summary)
        self.enabled = False
        return False


SolveBreakpoint("ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
                internal=True)
end

run
quit
