set pagination off
set confirm off
set print thread-events off
start
break ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)
continue

python
import gdb
import struct
from collections import Counter


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def i32(address):
    return struct.unpack("<i", read(address, 4))[0]


problem = int(gdb.parse_and_eval("$rsi"))
implementation = u64(problem)
program = None
for offset in range(0, 512, 8):
    try:
        candidate = u64(implementation + offset)
        parameter_candidate_begin = u64(candidate)
        parameter_candidate_end = u64(candidate + 8)
        residual_candidate_begin = u64(candidate + 24)
        residual_candidate_end = u64(candidate + 32)
    except (gdb.MemoryError, OverflowError):
        continue
    if (
        parameter_candidate_end >= parameter_candidate_begin
        and residual_candidate_end >= residual_candidate_begin
        and (parameter_candidate_end - parameter_candidate_begin) // 8 == 7993
        and (residual_candidate_end - residual_candidate_begin) // 8 == 5420
    ):
        program = candidate
        gdb.write(f"CLEAN_PROGRAM_POINTER implementation_offset={offset}\n")
        break
if program is None:
    raise gdb.GdbError("clean Ceres Program pointer was not found")
parameter_begin = u64(program)
parameter_end = u64(program + 8)
parameter_blocks = struct.unpack(
    f"<{(parameter_end - parameter_begin) // 8}Q",
    read(parameter_begin, parameter_end - parameter_begin),
)
parameter_indices = {pointer: index for index, pointer in enumerate(parameter_blocks)}
parameter_sizes = [i32(pointer + 8) for pointer in parameter_blocks]
gdb.write(
    f"CLEAN_PARAMETER_VECTOR count={len(parameter_blocks)} sizes_head="
    + ",".join(str(value) for value in parameter_sizes[:16])
    + " sizes_tail="
    + ",".join(str(value) for value in parameter_sizes[-16:])
    + "\n"
)

residual_begin = u64(program + 24)
residual_end = u64(program + 32)
residual_blocks = struct.unpack(
    f"<{(residual_end - residual_begin) // 8}Q",
    read(residual_begin, residual_end - residual_begin),
)
signatures = []
for block in residual_blocks:
    cost = u64(block)
    sizes_begin = u64(cost + 8)
    sizes_end = u64(cost + 16)
    sizes = struct.unpack(f"<{(sizes_end - sizes_begin) // 4}i", read(sizes_begin, sizes_end - sizes_begin))
    signatures.append((i32(cost + 32), sizes, u64(block + 8) != 0))
for signature, count in sorted(Counter(signatures).items(), key=lambda item: (-item[1], item[0])):
    residual_count, sizes, has_loss = signature
    gdb.write(
        f"CLEAN_RESIDUAL_SIGNATURE count={count} residuals={residual_count} "
        f"sizes={','.join(str(value) for value in sizes)} loss={int(has_loss)}\n"
    )
run_start = 0
for index in range(1, len(signatures) + 1):
    if index < len(signatures) and signatures[index] == signatures[run_start]:
        continue
    residual_count, sizes, has_loss = signatures[run_start]
    gdb.write(
        f"CLEAN_RESIDUAL_RUN begin={run_start} end={index} count={index-run_start} "
        f"residuals={residual_count} sizes={','.join(str(value) for value in sizes)} "
        f"loss={int(has_loss)}\n"
    )
    run_start = index
for index in (0, 2754, 2755, 2756, 2757, 2758, 2759, 5417, 5418, 5419):
    block = residual_blocks[index]
    cost = u64(block)
    sizes_begin = u64(cost + 8)
    sizes_end = u64(cost + 16)
    count = (sizes_end - sizes_begin) // 4
    connected = struct.unpack(f"<{count}Q", read(u64(block + 16), count * 8))
    gdb.write(
        f"CLEAN_RESIDUAL index={index} connected_indices="
        + ",".join(str(parameter_indices[pointer]) for pointer in connected)
        + "\n"
    )
end

quit
