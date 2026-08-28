set pagination off
set confirm off
set breakpoint pending on
set disable-randomization on
set print thread-events off

python
import hashlib
import json
import os
import pathlib
import struct

import gdb


out_dir = pathlib.Path(os.environ["GAMMA_JACOBIAN_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
input_path = pathlib.Path(os.environ["GAMMA_JACOBIAN_OVS"])
max_iterations = int(os.environ.get("GAMMA_JACOBIAN_MAX_ITERATIONS", "2"))
initial_radius = float(os.environ.get("GAMMA_JACOBIAN_INITIAL_RADIUS", "10000"))
summary_address = 0


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def write(address, payload):
    gdb.selected_inferior().write_memory(address, payload)


def u32(address):
    return struct.unpack("<I", read(address, 4))[0]


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def f64(address):
    return struct.unpack("<d", read(address, 8))[0]


def i32(address):
    return struct.unpack("<i", read(address, 4))[0]


def i64(address):
    return struct.unpack("<q", read(address, 8))[0]


def write_capture(name, payload):
    (out_dir / name).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def capture_summary(address):
    raw = read(address, 512)
    (out_dir / "solver_summary_512.bin").write_bytes(raw)
    begin, end = struct.unpack_from("<QQ", raw, 64)
    stride = 120
    if end < begin or (end - begin) % stride:
        raise gdb.GdbError(
            "unexpected IterationSummary vector: begin=%#x end=%#x" % (begin, end)
        )
    iteration_raw = read(begin, end - begin)
    (out_dir / "iteration_summaries.bin").write_bytes(iteration_raw)
    iterations = []
    for offset in range(0, len(iteration_raw), stride):
        iterations.append({
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
        })
    (out_dir / "iteration_summaries.json").write_text(
        json.dumps(iterations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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


class ExposureBreakpoint(gdb.Breakpoint):
    def stop(self):
        vector = int(gdb.parse_and_eval("$rsi"))
        begin = u64(vector)
        end = u64(vector + 8)
        before = read(begin, end - begin)
        effective = input_path.read_bytes()
        if len(effective) != len(before):
            raise gdb.GdbError("OVS size mismatch %d != %d" % (len(effective), len(before)))
        write(begin, effective)
        (out_dir / "effective_exposure_ovs.bin").write_bytes(effective)
        (out_dir / "ovs_capture.json").write_text(
            json.dumps({
                "records": len(effective) // 40,
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "effective_sha256": hashlib.sha256(effective).hexdigest(),
                "source": str(input_path),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return False


class SolveReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, address):
        super().__init__(frame, internal=True)
        self.address = address

    def stop(self):
        capture_summary(self.address)
        return False


class SolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        global summary_address
        options = int(gdb.parse_and_eval("$rdi"))
        summary_address = int(gdb.parse_and_eval("$rdx"))
        before = read(options, 488)
        (out_dir / "solver_options_before.bin").write_bytes(before)
        write(options + 104, struct.pack("<i", max_iterations))
        write(options + 120, struct.pack("<i", 1))
        write(options + 128, struct.pack("<d", initial_radius))
        # Ceres 2.1 layout: suppress progress output without changing numerics.
        write(options + 364, struct.pack("<i", 1))
        write(options + 368, b"\x00")
        effective = read(options, 488)
        (out_dir / "solver_options_effective.bin").write_bytes(effective)
        (out_dir / "solver_overrides.json").write_text(
            json.dumps({
                "max_iterations_before": struct.unpack_from("<i", before, 104)[0],
                "max_iterations_effective": max_iterations,
                "threads_before": struct.unpack_from("<i", before, 120)[0],
                "threads_effective": 1,
                "initial_radius_before": struct.unpack_from("<d", before, 128)[0],
                "initial_radius_effective": initial_radius,
                "options_before_sha256": hashlib.sha256(before).hexdigest(),
                "options_effective_sha256": hashlib.sha256(effective).hexdigest(),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        SolveReturnBreakpoint(gdb.newest_frame(), summary_address)
        self.enabled = False
        return False


class ModelBreakpoint(gdb.Breakpoint):
    def stop(self):
        models = 0x555555EF7960
        node = u64(models + 0x10)
        values = {}
        while node:
            raw_view = u32(node + 8)
            gamma = u64(node + 0x10)
            view = 4 * (raw_view >> 16) + ((raw_view >> 8) & 0xFF)
            values[view] = (f64(gamma), f64(gamma + 8))
            node = u64(node)
        with (out_dir / "gamma_models.txt").open("w", encoding="utf-8") as output:
            output.write("# view gain exponent\n")
            for view in sorted(values):
                gain, exponent = values[view]
                output.write("%d %.17g %.17g\n" % (view, gain, exponent))
        self.enabled = False
        return True


class IdentityEvaluationReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, minimizer):
        super().__init__(frame, internal=True)
        self.minimizer = minimizer

    def stop(self):
        minimizer = self.minimizer
        state = u64(minimizer + 0x218)
        state_size = i64(minimizer + 0x220)
        residuals = u64(minimizer + 0x228)
        residual_count = i64(minimizer + 0x230)
        gradient = u64(minimizer + 0x238)
        gradient_count = i64(minimizer + 0x240)
        jacobian = u64(minimizer + 0x170)
        jacobian_rows = i32(jacobian + 0x8)
        jacobian_cols = i32(jacobian + 0xC)
        jacobian_nonzeros = i32(jacobian + 0x10)
        jacobian_values = u64(jacobian + 0x18)
        scaling = u64(minimizer + 0x2B8)
        scaling_count = i64(minimizer + 0x2C0)
        payloads = {
            "identity_state.bin": read(state, state_size * 8),
            "identity_residual_b.bin": read(residuals, residual_count * 8),
            "identity_gradient.bin": read(gradient, gradient_count * 8),
            "identity_jacobian_scaled_A.bin": read(
                jacobian_values, jacobian_nonzeros * 8
            ),
            "identity_jacobian_scaling.bin": read(scaling, scaling_count * 8),
        }
        hashes = {name: write_capture(name, payload) for name, payload in payloads.items()}
        metadata = {
            "minimizer_address": minimizer,
            "state_count": state_size,
            "residual_count": residual_count,
            "gradient_count": gradient_count,
            "jacobian_address": jacobian,
            "jacobian_rows": jacobian_rows,
            "jacobian_cols": jacobian_cols,
            "jacobian_nonzeros": jacobian_nonzeros,
            "scaling_count": scaling_count,
            "sha256": hashes,
        }
        if not (
            state_size == gradient_count == scaling_count == jacobian_cols == 272
            and residual_count == jacobian_rows
        ):
            raise gdb.GdbError("unexpected identity evaluation dimensions: %s" % metadata)
        (out_dir / "identity_evaluation.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return False


class IdentityEvaluationBreakpoint(gdb.Breakpoint):
    def stop(self):
        minimizer = int(gdb.parse_and_eval("$rdi"))
        iteration = i32(minimizer + 0x190)
        if iteration == 0:
            IdentityEvaluationReturnBreakpoint(gdb.newest_frame(), minimizer)
            self.enabled = False
        return False


class CgnrPostLeftMultiplyBreakpoint(gdb.Breakpoint):
    def __init__(self, address, count):
        super().__init__("*0x%x" % address, internal=True, temporary=True)
        self.count = count

    def stop(self):
        rhs = u64(int(gdb.parse_and_eval("$rsp")) + 0x18)
        write_capture("identity_cgnr_rhs_Atb.bin", read(rhs, self.count * 8))
        return False


class CgnrReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, output, count):
        super().__init__(frame, internal=True)
        self.output = output
        self.count = count

    def stop(self):
        write_capture("identity_cgnr_solution_y.bin", read(self.output, self.count * 8))
        return False


class FirstCgnrBreakpoint(gdb.Breakpoint):
    def stop(self):
        # LinearSolver::Summary is returned through a hidden sret pointer in
        # RDI, shifting the five explicit arguments by one register.
        jacobian = int(gdb.parse_and_eval("$rdx"))
        residuals = int(gdb.parse_and_eval("$rcx"))
        per_solve_options = int(gdb.parse_and_eval("$r8"))
        output = int(gdb.parse_and_eval("$r9"))
        rows = i32(jacobian + 0x8)
        cols = i32(jacobian + 0xC)
        nonzeros = i32(jacobian + 0x10)
        values = u64(jacobian + 0x18)
        diagonal = u64(per_solve_options)
        payloads = {
            "identity_cgnr_A.bin": read(values, nonzeros * 8),
            "identity_cgnr_b.bin": read(residuals, rows * 8),
            "identity_cgnr_D.bin": read(diagonal, cols * 8),
        }
        hashes = {name: write_capture(name, payload) for name, payload in payloads.items()}
        entry = int(gdb.parse_and_eval("$pc"))
        CgnrPostLeftMultiplyBreakpoint(entry + 0x11C, cols)
        CgnrReturnBreakpoint(gdb.newest_frame(), output, cols)
        (out_dir / "identity_cgnr.json").write_text(
            json.dumps({
                "entry_address": entry,
                "post_left_multiply_address": entry + 0x11C,
                "jacobian_rows": rows,
                "jacobian_cols": cols,
                "jacobian_nonzeros": nonzeros,
                "diagonal_address": diagonal,
                "output_address": output,
                "sha256_at_entry": hashes,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return False


class FirstJointCostReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, residuals, jacobians, residual_count, block_sizes, parameters):
        super().__init__(frame, internal=True)
        self.residuals = residuals
        self.jacobians = jacobians
        self.residual_count = residual_count
        self.block_sizes = block_sizes
        self.parameters = parameters

    def stop(self):
        residual_payload = read(self.residuals, self.residual_count * 8)
        parameter_payload = bytearray()
        jacobian_payload = bytearray()
        block_offsets = []
        for block, size in enumerate(self.block_sizes):
            parameter_pointer = u64(self.parameters + block * 8)
            parameter_payload.extend(read(parameter_pointer, size * 8))
            jacobian_pointer = u64(self.jacobians + block * 8)
            if not jacobian_pointer:
                raise gdb.GdbError("first Joint jacobian block %d is null" % block)
            begin = len(jacobian_payload) // 8
            jacobian_payload.extend(read(jacobian_pointer, self.residual_count * size * 8))
            block_offsets.append({
                "block": block,
                "parameter_size": size,
                "jacobian_begin": begin,
                "jacobian_end": len(jacobian_payload) // 8,
            })
        hashes = {
            "first_joint_raw_parameters.bin": write_capture(
                "first_joint_raw_parameters.bin", bytes(parameter_payload)
            ),
            "first_joint_raw_residual.bin": write_capture(
                "first_joint_raw_residual.bin", residual_payload
            ),
            "first_joint_raw_jacobian.bin": write_capture(
                "first_joint_raw_jacobian.bin", bytes(jacobian_payload)
            ),
        }
        (out_dir / "first_joint_raw.json").write_text(
            json.dumps({
                "residual_count": self.residual_count,
                "parameter_block_count": len(self.block_sizes),
                "parameter_block_sizes": self.block_sizes,
                "jacobian_scalar_count": len(jacobian_payload) // 8,
                "block_offsets": block_offsets,
                "sha256": hashes,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return False


class FirstJointCostEvaluateBreakpoint(gdb.Breakpoint):
    def stop(self):
        cost = int(gdb.parse_and_eval("$rdi"))
        parameters = int(gdb.parse_and_eval("$rsi"))
        residuals = int(gdb.parse_and_eval("$rdx"))
        jacobians = int(gdb.parse_and_eval("$rcx"))
        residual_count = i32(cost + 0x20)
        sizes_begin = u64(cost + 0x8)
        sizes_end = u64(cost + 0x10)
        block_sizes = [i32(address) for address in range(sizes_begin, sizes_end, 4)]
        if not (2 <= residual_count <= 5 and block_sizes == [2] * residual_count):
            raise gdb.GdbError(
                "unexpected first Joint dimensions residuals=%d blocks=%s"
                % (residual_count, block_sizes)
            )
        if not jacobians:
            return False
        FirstJointCostReturnBreakpoint(
            gdb.newest_frame(), residuals, jacobians, residual_count, block_sizes, parameters
        )
        self.enabled = False
        return False


class FirstJointAddResidualBreakpoint(gdb.Breakpoint):
    def stop(self):
        cost = int(gdb.parse_and_eval("$rsi"))
        vtable = u64(cost)
        evaluate = u64(vtable + 0x10)
        (out_dir / "first_joint_evaluate_address.json").write_text(
            json.dumps({
                "cost_function_address": cost,
                "vtable_address": vtable,
                "evaluate_address": evaluate,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        FirstJointCostEvaluateBreakpoint("*0x%x" % evaluate, internal=True)
        self.enabled = False
        return False


ExposureBreakpoint("*0x55555573ea70", internal=True)
SolveBreakpoint("*0x555555a7e890", internal=True)
ModelBreakpoint("*0x55555573936b", internal=True)
IdentityEvaluationBreakpoint("*0x555555aa76b0", internal=True)
FirstCgnrBreakpoint("*0x555555c9abf0", internal=True)
FirstJointAddResidualBreakpoint("*0x555555a70680", internal=True)
end

run
quit
