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
max_iterations = int(os.environ.get("GAMMA_JACOBIAN_MAX_ITERATIONS", "2"))
initial_radius = float(os.environ.get("GAMMA_JACOBIAN_INITIAL_RADIUS", "10000"))
vendor_max1_path = pathlib.Path(os.environ["GAMMA_JACOBIAN_VENDOR_MAX1"])
clean_max1_path = pathlib.Path(os.environ["GAMMA_JACOBIAN_CLEAN_MAX1"])


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def write(address, payload):
    gdb.selected_inferior().write_memory(address, payload)


def i32(address):
    return struct.unpack("<i", read(address, 4))[0]


def i64(address):
    return struct.unpack("<q", read(address, 8))[0]


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def write_capture(name, payload):
    (out_dir / name).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_models(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        view_text, gain_text, exponent_text = line.split()
        result[int(view_text)] = (float(gain_text), float(exponent_text))
    if sorted(result) != list(range(136)):
        raise gdb.GdbError("expected views 0..135 in %s" % path)
    return result


vendor_max1 = load_models(vendor_max1_path)
clean_max1 = load_models(clean_max1_path)


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


class SolveReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, summary_address):
        super().__init__(frame, internal=True)
        self.summary_address = summary_address

    def stop(self):
        capture_summary(self.summary_address)
        return False


class CeresSolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        options = int(gdb.parse_and_eval("$rdi"))
        summary = int(gdb.parse_and_eval("$rdx"))
        before = read(options, 488)
        (out_dir / "solver_options_before.bin").write_bytes(before)
        write(options + 104, struct.pack("<i", max_iterations))
        write(options + 120, struct.pack("<i", 1))
        write(options + 128, struct.pack("<d", initial_radius))
        effective = read(options, 488)
        (out_dir / "solver_options_effective.bin").write_bytes(effective)
        metadata = {
            "max_iterations_before": struct.unpack_from("<i", before, 104)[0],
            "max_iterations_effective": max_iterations,
            "threads_before": struct.unpack_from("<i", before, 120)[0],
            "threads_effective": 1,
            "initial_radius_before": struct.unpack_from("<d", before, 128)[0],
            "initial_radius_effective": initial_radius,
            "options_before_sha256": hashlib.sha256(before).hexdigest(),
            "options_effective_sha256": hashlib.sha256(effective).hexdigest(),
        }
        (out_dir / "solver_overrides.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        SolveReturnBreakpoint(gdb.newest_frame(), summary)
        self.enabled = False
        return False


class ExposureSolveBreakpoint(gdb.Breakpoint):
    """Round the old Ceres-2.1 sidecar binary's double Scene weights to vendor f32."""

    def stop(self):
        argument_address = int(gdb.parse_and_eval("$rsi"))
        stride = 16
        candidates = []
        for relative_offset in range(-48, 145, 24):
            vector_address = argument_address + relative_offset
            try:
                begin, end, capacity = struct.unpack("<QQQ", read(vector_address, 24))
                if end < begin or capacity < end or end - begin != 136 * stride:
                    continue
                views = []
                valid = True
                for index in range(136):
                    item = read(begin + index * stride, stride)
                    view = struct.unpack_from("<i", item, 0)[0]
                    low, high = struct.unpack_from("<BB", item, 4)
                    if not (0 <= view < 136 and low < high):
                        valid = False
                        break
                    views.append(view)
                if valid and sorted(views) == list(range(136)):
                    candidates.append((relative_offset, vector_address, begin, end))
            except gdb.MemoryError:
                continue
        if not candidates:
            # The later native-f32 sidecar binary stores Scene as a 12-byte
            # item (int view, three bytes, one float) and needs no runtime
            # rounding.  Detect that layout explicitly rather than silently
            # skipping a failed double-layout probe.
            native_candidates = []
            native_stride = 12
            for relative_offset in range(-48, 145, 24):
                vector_address = argument_address + relative_offset
                try:
                    begin, end, capacity = struct.unpack("<QQQ", read(vector_address, 24))
                    if (
                        end < begin
                        or capacity < end
                        or end - begin != 136 * native_stride
                    ):
                        continue
                    views = []
                    valid = True
                    for index in range(136):
                        item = read(begin + index * native_stride, native_stride)
                        view = struct.unpack_from("<i", item, 0)[0]
                        low, high = struct.unpack_from("<BB", item, 4)
                        weight = struct.unpack_from("<f", item, 8)[0]
                        if not (0 <= view < 136 and low < high and weight >= 0.0):
                            valid = False
                            break
                        views.append(view)
                    if valid and sorted(views) == list(range(136)):
                        native_candidates.append(
                            (relative_offset, vector_address, begin, end)
                        )
                except gdb.MemoryError:
                    continue
            if len(native_candidates) != 1:
                raise gdb.GdbError(
                    "expected one double or native-float Scene vector; found %d/%d"
                    % (len(candidates), len(native_candidates))
                )
            relative_offset, vector_address, begin, end = native_candidates[0]
            (out_dir / "scene_weight_patch.json").write_text(
                json.dumps({
                    "mode": "native_float_no_patch",
                    "argument_address": argument_address,
                    "scene_vector_address": vector_address,
                    "scene_vector_relative_offset": relative_offset,
                    "scene_begin": begin,
                    "scene_end": end,
                    "scene_stride": native_stride,
                    "changed_count": 0,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.enabled = False
            return False
        if len(candidates) != 1:
            raise gdb.GdbError("expected one double Scene vector, found %d" % len(candidates))
        relative_offset, vector_address, begin, end = candidates[0]
        changed = 0
        maximum_delta = 0.0
        rows = []
        for index in range(136):
            item_address = begin + index * stride
            item = read(item_address, stride)
            view = struct.unpack_from("<i", item, 0)[0]
            original = struct.unpack_from("<d", item, 8)[0]
            rounded = float(struct.unpack("<f", struct.pack("<f", original))[0])
            delta = rounded - original
            if rounded != original:
                write(item_address + 8, struct.pack("<d", rounded))
                changed += 1
            maximum_delta = max(maximum_delta, abs(delta))
            rows.append({"index": index, "view": view, "before": original, "after": rounded})
        (out_dir / "scene_weight_patch.json").write_text(
            json.dumps({
                "argument_address": argument_address,
                "scene_vector_address": vector_address,
                "scene_vector_relative_offset": relative_offset,
                "scene_begin": begin,
                "scene_end": end,
                "changed_count": changed,
                "maximum_abs_delta": maximum_delta,
                "rows": rows,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return False


class Iter1StateInjectionBreakpoint(gdb.Breakpoint):
    """Replace clean iter1 x with vendor iter1 x inside the same minimizer."""

    def stop(self):
        minimizer = int(gdb.parse_and_eval("$rdi"))
        new_evaluation_point = int(gdb.parse_and_eval("$esi"))
        iteration = struct.unpack("<i", read(minimizer + 0x190, 4))[0]
        if iteration != 1 or new_evaluation_point != 0:
            return False

        state = struct.unpack("<Q", read(minimizer + 0x218, 8))[0]
        state_size = struct.unpack("<q", read(minimizer + 0x220, 8))[0]
        if state_size != 272:
            raise gdb.GdbError("expected 272 state scalars, got %d" % state_size)
        before = read(state, state_size * 8)

        # Parameter blocks are two adjacent doubles but follow Ceres first-use
        # order, not numeric view order.  Resolve the order without guessing by
        # requiring every pre-injection pair to match the frozen clean max1
        # model bit-for-bit.
        pair_to_view = {
            struct.pack("<dd", gain, exponent): view
            for view, (gain, exponent) in clean_max1.items()
        }
        if len(pair_to_view) != 136:
            raise gdb.GdbError("clean max1 model pairs are not unique")
        mapped_views = []
        after = bytearray(before)
        rows = []
        for block in range(136):
            pair = before[block * 16 : (block + 1) * 16]
            if pair not in pair_to_view:
                gain, exponent = struct.unpack("<dd", pair)
                raise gdb.GdbError(
                    "state block %d does not exactly match clean max1: %.17g %.17g"
                    % (block, gain, exponent)
                )
            view = pair_to_view[pair]
            mapped_views.append(view)
            clean_gain, clean_exponent = struct.unpack("<dd", pair)
            vendor_gain, vendor_exponent = vendor_max1[view]
            replacement = struct.pack("<dd", vendor_gain, vendor_exponent)
            after[block * 16 : (block + 1) * 16] = replacement
            rows.append({
                "block": block,
                "view": view,
                "clean_gain": clean_gain,
                "clean_exponent": clean_exponent,
                "vendor_gain": vendor_gain,
                "vendor_exponent": vendor_exponent,
                "gain_delta": vendor_gain - clean_gain,
                "exponent_delta": vendor_exponent - clean_exponent,
            })
        if sorted(mapped_views) != list(range(136)):
            raise gdb.GdbError("state mapping is not a permutation of views 0..135")
        write(state, bytes(after))
        (out_dir / "iter1_state_before.bin").write_bytes(before)
        (out_dir / "iter1_state_vendor_injected.bin").write_bytes(bytes(after))
        (out_dir / "iter1_state_injection.json").write_text(
            json.dumps({
                "minimizer_address": minimizer,
                "state_address": state,
                "state_scalars": state_size,
                "iteration": iteration,
                "new_evaluation_point": new_evaluation_point,
                "clean_max1_path": str(clean_max1_path),
                "clean_max1_sha256": hashlib.sha256(clean_max1_path.read_bytes()).hexdigest(),
                "vendor_max1_path": str(vendor_max1_path),
                "vendor_max1_sha256": hashlib.sha256(vendor_max1_path.read_bytes()).hexdigest(),
                "state_before_sha256": hashlib.sha256(before).hexdigest(),
                "state_after_sha256": hashlib.sha256(after).hexdigest(),
                "mapped_views": mapped_views,
                "rows": rows,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return False


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
        payload = read(rhs, self.count * 8)
        write_capture("identity_cgnr_rhs_Atb.bin", payload)
        return False


class CgnrReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, output, count):
        super().__init__(frame, internal=True)
        self.output = output
        self.count = count

    def stop(self):
        payload = read(self.output, self.count * 8)
        write_capture("identity_cgnr_solution_y.bin", payload)
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


ExposureSolveBreakpoint("solveExposureProblem", internal=True)
CeresSolveBreakpoint(
    "ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
    internal=True,
)
Iter1StateInjectionBreakpoint(
    "ceres::internal::TrustRegionMinimizer::EvaluateGradientAndJacobian(bool)",
    internal=True,
)
IdentityEvaluationBreakpoint(
    "ceres::internal::TrustRegionMinimizer::EvaluateGradientAndJacobian(bool)",
    internal=True,
)
FirstCgnrBreakpoint(
    "ceres::internal::CgnrSolver::SolveImpl(ceres::internal::BlockSparseMatrix*, double const*, ceres::internal::LinearSolver::PerSolveOptions const&, double*)",
    internal=True,
)
FirstJointAddResidualBreakpoint(
    "ceres::Problem::AddResidualBlock(ceres::CostFunction*, ceres::LossFunction*, std::vector<double*, std::allocator<double*> > const&)",
    internal=True,
)
end

run
quit
