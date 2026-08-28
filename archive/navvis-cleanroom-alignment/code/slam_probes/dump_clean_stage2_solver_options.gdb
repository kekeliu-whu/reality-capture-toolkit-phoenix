set pagination off
set confirm off
set print thread-events off
start
break ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)
continue

python
import gdb

options = int(gdb.parse_and_eval("$rdi"))
raw = gdb.selected_inferior().read_memory(options, 488).tobytes()
gdb.write("SOLVER_OPTIONS_HEX " + raw.hex() + "\n")
end

quit
