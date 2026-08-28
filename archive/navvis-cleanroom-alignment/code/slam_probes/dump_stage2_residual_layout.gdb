set pagination off
set confirm off
set print thread-events off
starti

python
import gdb
import struct
from collections import Counter


EXECUTABLE = "/opt/NavVis/slam/lib/surveyor_ros/compute_trajectories"
CERES_SOLVE_OFFSET = 0x324E40
# 2756 pose-graph constraints + 2659 consecutive-node IMU factors + five
# calibration priors in the installed optimization_problem_imu_intrinsics path.
EXPECTED_RESIDUAL_BLOCKS = 2756 + 2659 + 5


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def i32(address):
    return struct.unpack("<i", read(address, 4))[0]


def executable_base():
    pid = gdb.selected_inferior().pid
    with open(f"/proc/{pid}/maps", "r", encoding="ascii") as stream:
        for line in stream:
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or fields[5].strip() != EXECUTABLE:
                continue
            start, _ = (int(value, 16) for value in fields[0].split("-"))
            return start - int(fields[2], 16)
    raise gdb.GdbError("compute_trajectories executable mapping was not found")


def vector_count(address):
    begin, end, capacity = struct.unpack("<QQQ", read(address, 24))
    if begin == 0 or end < begin or capacity < end or (end - begin) % 8:
        return None
    count = (end - begin) // 8
    if count > 20000:
        return None
    return begin, count


def dump_cost_function(cost):
    # CostFunction has a vptr followed by vector<int32_t> parameter sizes and
    # then num_residuals in this statically linked Ceres build.
    begin = u64(cost + 8)
    end = u64(cost + 16)
    num_residuals = i32(cost + 32)
    if end < begin or (end - begin) % 4:
        gdb.write(f" cost={cost:#x} residuals={num_residuals} sizes=INVALID")
        return
    sizes = struct.unpack(f"<{(end - begin) // 4}i", read(begin, end - begin))
    gdb.write(
        f" cost={cost:#x} residuals={num_residuals} parameter_sizes="
        + ",".join(str(value) for value in sizes)
    )


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{executable_base() + CERES_SOLVE_OFFSET:#x}", internal=True)

    def stop(self):
        problem = int(gdb.parse_and_eval("$rsi"))
        implementation = u64(problem)
        program = u64(implementation + 160)
        gdb.write(
            f"CERES_OBJECT problem={problem:#x} implementation={implementation:#x} "
            f"program={program:#x}\n"
        )

        residual_vector = None
        for offset in range(0, 512, 8):
            candidate = vector_count(program + offset)
            if candidate is None:
                continue
            begin, count = candidate
            if count:
                gdb.write(
                    f"PROGRAM_VECTOR offset={offset} begin={begin:#x} count={count}\n"
                )
            if count == EXPECTED_RESIDUAL_BLOCKS:
                residual_vector = candidate

        if residual_vector is None:
            raise gdb.GdbError("the expected residual-block vector was not found")

        begin, count = residual_vector
        residuals = struct.unpack(f"<{count}Q", read(begin, count * 8))
        parameter_vector_begin = u64(program)
        parameter_vector_end = u64(program + 8)
        parameter_blocks_in_program = struct.unpack(
            f"<{(parameter_vector_end - parameter_vector_begin) // 8}Q",
            read(parameter_vector_begin, parameter_vector_end - parameter_vector_begin),
        )
        parameter_indices = {
            pointer: index for index, pointer in enumerate(parameter_blocks_in_program)
        }
        signatures = []
        for block in residuals:
            cost = u64(block)
            sizes_begin = u64(cost + 8)
            sizes_end = u64(cost + 16)
            size_count = (sizes_end - sizes_begin) // 4
            sizes = struct.unpack(
                f"<{size_count}i", read(sizes_begin, size_count * 4)
            )
            signatures.append(
                (i32(cost + 32), sizes, u64(block + 8) != 0)
            )
        for signature, signature_count in sorted(
            Counter(signatures).items(), key=lambda item: (-item[1], item[0])
        ):
            residuals_count, sizes, has_loss = signature
            gdb.write(
                f"RESIDUAL_SIGNATURE count={signature_count} residuals={residuals_count} "
                f"sizes={','.join(str(value) for value in sizes)} "
                f"loss={int(has_loss)}\n"
            )
        run_start = 0
        for index in range(1, len(signatures) + 1):
            if index < len(signatures) and signatures[index] == signatures[run_start]:
                continue
            residuals_count, sizes, has_loss = signatures[run_start]
            gdb.write(
                f"RESIDUAL_RUN begin={run_start} end={index} count={index-run_start} "
                f"residuals={residuals_count} sizes="
                f"{','.join(str(value) for value in sizes)} loss={int(has_loss)}\n"
            )
            run_start = index
        selected = (0, 2754, 2755, 2756, 2757, 2758, 2759, 5413, 5414, 5415, 5416, 5417, 5418, 5419)
        for index in selected:
            block = residuals[index]
            words = struct.unpack("<8Q", read(block, 64))
            gdb.write(
                f"RESIDUAL index={index} block={block:#x} raw="
                + " ".join(f"{value:#x}" for value in words)
            )
            # The installed ResidualBlock begins with CostFunction*,
            # LossFunction*, ParameterBlock** and index.  The parameter count
            # is carried by CostFunction::parameter_block_sizes().
            cost = words[0]
            sizes_begin = u64(cost + 8)
            sizes_end = u64(cost + 16)
            num_parameters = (sizes_end - sizes_begin) // 4
            parameter_array = u64(block + 16)
            block_index = i32(block + 24)
            gdb.write(
                f" decoded_num_parameters={num_parameters} decoded_index={block_index}"
            )
            if 0 < num_parameters < 32 and parameter_array:
                parameter_blocks = struct.unpack(
                    f"<{num_parameters}Q", read(parameter_array, num_parameters * 8)
                )
                parameter_sizes = [i32(pointer + 8) for pointer in parameter_blocks]
                gdb.write(
                    " connected_sizes=" + ",".join(str(value) for value in parameter_sizes)
                )
                gdb.write(
                    " connected_indices="
                    + ",".join(str(parameter_indices[pointer]) for pointer in parameter_blocks)
                )
            dump_cost_function(cost)
            loss = u64(block + 8)
            if loss:
                loss_words = struct.unpack("<4Q", read(loss, 32))
                loss_doubles = struct.unpack("<4d", read(loss, 32))
                gdb.write(
                    " loss_raw=" + " ".join(f"{value:#x}" for value in loss_words)
                    + " loss_doubles="
                    + ",".join(format(value, ".17g") for value in loss_doubles)
                )
            gdb.write("\n")

        gdb.execute("quit")
        return False


SolveEntry()
end

continue
