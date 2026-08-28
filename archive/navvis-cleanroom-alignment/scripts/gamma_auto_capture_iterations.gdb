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

out_dir = pathlib.Path(os.environ["GAMMA_AUTO_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
report = (out_dir / "capture.txt").open("w", encoding="utf-8", buffering=1)
enable_order = os.environ.get("GAMMA_AUTO_CAPTURE_ORDER", "0") == "1"
input_path = os.environ.get("GAMMA_AUTO_EXPOSURE_INPUT", "")
solver_threads_override = os.environ.get("GAMMA_AUTO_SOLVER_THREADS", "")
max_iterations_override = os.environ.get("GAMMA_MAX_ITERATIONS", "")

seen = []
seen_set = set()
pointer_to_raw_view = {}
pending_raw_view = None
residual_index = 0
vector_residuals = 0
scalar_residuals = 0
scalar_residual_dimension = 0
first_dynamic_helper = None
first_scene_helper = None
dynamic_inputs = []
scene_input = None
summary_pointer = 0


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u32(address):
    return struct.unpack("<I", read(address, 4))[0]


def i32(address):
    return struct.unpack("<i", read(address, 4))[0]


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def f64(address):
    return struct.unpack("<d", read(address, 8))[0]


class ExposureBreakpoint(gdb.Breakpoint):
    def stop(self):
        vector = int(gdb.parse_and_eval("$rsi"))
        begin = u64(vector)
        end = u64(vector + 8)
        actual = read(begin, end - begin)
        report.write(
            "EXPOSURE_OVS_BEFORE records=%d bytes=%d sha256=%s\n"
            % ((end - begin) // 40, end - begin, hashlib.sha256(actual).hexdigest())
        )
        effective = actual
        if input_path:
            effective = pathlib.Path(input_path).read_bytes()
            if len(effective) != len(actual):
                raise RuntimeError(
                    "fixed exposure OVS size mismatch: %d != %d"
                    % (len(effective), len(actual))
                )
            gdb.selected_inferior().write_memory(begin, effective)
            report.write(
                "EXPOSURE_OVS_INJECTED path=%s sha256=%s\n"
                % (input_path, hashlib.sha256(effective).hexdigest())
            )
        (out_dir / "effective_exposure_ovs.bin").write_bytes(effective)
        self.enabled = False
        return False


class AddResidualBreakpoint(gdb.Breakpoint):
    def __init__(self, specification, vector_form):
        super().__init__(specification, internal=True)
        self.vector_form = vector_form

    def stop(self):
        global pending_raw_view, residual_index, vector_residuals, scalar_residuals
        global scalar_residual_dimension
        if not enable_order:
            self.enabled = False
            return False
        if self.vector_form:
            vector_residuals += 1
            vector_address = int(gdb.parse_and_eval("$rcx"))
            blocks_address, blocks_end = struct.unpack("<QQ", read(vector_address, 16))
            count = (blocks_end - blocks_address) // 8
        else:
            scalar_residuals += 1
            count = int(gdb.parse_and_eval("$r8"))
            blocks_address = int(gdb.parse_and_eval("$rcx"))
        cost_function = int(gdb.parse_and_eval("$rsi"))
        scalar_residual_dimension += i32(cost_function + 32)
        if 0 < count <= 10 and blocks_address:
            blocks = struct.unpack("<" + "Q" * count, read(blocks_address, 8 * count))
            for pointer in blocks:
                if pointer not in seen_set:
                    seen_set.add(pointer)
                    seen.append((pointer, residual_index))
            if pending_raw_view is not None and count == 1:
                pointer_to_raw_view[blocks[0]] = pending_raw_view
                pending_raw_view = None
        residual_index += 1
        return False


class DynamicRangeBreakpoint(gdb.Breakpoint):
    def stop(self):
        global pending_raw_view, first_dynamic_helper
        if not enable_order:
            self.enabled = False
            return False
        pending_raw_view = int(gdb.parse_and_eval("$esi")) & 0xFFFFFFFF
        low = int(gdb.parse_and_eval("$edx")) & 0xFF
        high = int(gdb.parse_and_eval("$ecx")) & 0xFF
        scale = float(gdb.parse_and_eval("$xmm0.v2_double[0]"))
        dynamic_inputs.append((pending_raw_view, low, high, scale))
        if first_dynamic_helper is None:
            first_dynamic_helper = residual_index
        return False


class SceneRangeBreakpoint(gdb.Breakpoint):
    def stop(self):
        global first_scene_helper, scene_input
        if not enable_order:
            self.enabled = False
            return False
        if first_scene_helper is None:
            first_scene_helper = residual_index
        vector = int(gdb.parse_and_eval("$rsi"))
        begin = u64(vector)
        end = u64(vector + 8)
        scale = float(gdb.parse_and_eval("$xmm0.v2_double[0]"))
        scene_input = (begin, end, scale, read(begin, end - begin))
        return False


class SolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        global summary_pointer
        options = int(gdb.parse_and_eval("$rdi"))
        summary_pointer = int(gdb.parse_and_eval("$rdx"))
        raw_before = read(options, 504)
        (out_dir / "solver_options_before_504.bin").write_bytes(raw_before)
        original_threads = struct.unpack_from("<i", raw_before, 120)[0]
        if solver_threads_override:
            effective_threads = int(solver_threads_override)
            gdb.selected_inferior().write_memory(
                options + 120, struct.pack("<i", effective_threads)
            )
            report.write(
                "SOLVER_THREADS_OVERRIDE original=%d effective=%d\n"
                % (original_threads, effective_threads)
            )
        if max_iterations_override:
            gdb.selected_inferior().write_memory(
                options + 104, struct.pack("<i", int(max_iterations_override))
            )
            report.write(
                "MAX_ITERATIONS_OVERRIDE original=%d effective=%d\n"
                % (struct.unpack_from("<i", raw_before, 104)[0], int(max_iterations_override))
            )
        raw = read(options, 504)
        (out_dir / "solver_options_effective_504.bin").write_bytes(raw)
        report.write("CERES_SOLVE options=0x%x summary=0x%x\n" % (options, summary_pointer))
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
        ]
        for name, offset, code in fields:
            value = struct.unpack_from("<" + code, raw, offset)[0]
            report.write("OPTION %s offset=%d value=%s\n" % (name, offset, value))
        # Fields after offset 212 distinguish the Ceres 2.0 and 2.2 layouts.
        for version, offsets in (
            ("ceres20", (("min_linear", 344, "i"), ("max_linear", 348, "i"),
                          ("eta", 352, "d"), ("jacobi", 360, "B"),
                          ("logging", 364, "i"), ("progress", 368, "B"))),
            ("ceres22", (("min_linear", 320, "i"), ("max_linear", 324, "i"),
                          ("eta", 344, "d"), ("jacobi", 352, "B"),
                          ("logging", 384, "i"), ("progress", 388, "B"))),
        ):
            values = []
            for name, offset, code in offsets:
                values.append("%s=%s" % (name, struct.unpack_from("<" + code, raw, offset)[0]))
            report.write("OPTION_LAYOUT %s %s\n" % (version, " ".join(values)))
        report.write(
            "RESIDUAL_ORDER total=%d vector=%d scalar=%d scalar_dimension=%d "
            "parameters=%d mapped=%d first_dynamic=%s first_scene=%s\n"
            % (residual_index, vector_residuals, scalar_residuals,
               scalar_residual_dimension, len(seen), len(pointer_to_raw_view),
               first_dynamic_helper, first_scene_helper)
        )
        with (out_dir / "dynamic_inputs.tsv").open("w", encoding="utf-8") as output:
            output.write("raw_view\tview\tlow\thigh\tloss_scale\n")
            for raw_view, low, high, scale in dynamic_inputs:
                view = 4 * (raw_view >> 16) + ((raw_view >> 8) & 0xFF)
                output.write("%d\t%d\t%d\t%d\t%.17g\n" %
                             (raw_view, view, low, high, scale))
        if scene_input is not None:
            begin, end, scale, scene_bytes = scene_input
            (out_dir / "scene_ranges.bin").write_bytes(scene_bytes)
            report.write("SCENE_INPUT bytes=%d scale=%.17g\n" % (end - begin, scale))
        order_path = out_dir / "parameter_first_use.tsv"
        with order_path.open("w", encoding="utf-8") as output:
            output.write("order\tpointer\tfirst_residual\traw_view\tview\n")
            for order, (pointer, first_residual) in enumerate(seen):
                raw_view = pointer_to_raw_view.get(pointer, -1)
                view = -1 if raw_view < 0 else 4 * (raw_view >> 16) + ((raw_view >> 8) & 0xFF)
                output.write(
                    "%d\t0x%x\t%d\t%d\t%d\n"
                    % (order, pointer, first_residual, raw_view, view)
                )
        self.enabled = False
        return False


class ModelBreakpoint(gdb.Breakpoint):
    def stop(self):
        # 0x1f7450 returns the process-global model container at static
        # 0x9a3960.  Read it directly: inferior function calls from a Python
        # breakpoint can resume a worker thread and make the capture racy.
        models = 0x555555EF7960
        node = u64(models + 0x10)
        values = {}
        while node:
            raw_view = u32(node + 8)
            gamma = u64(node + 0x10)
            view = 4 * (raw_view >> 16) + ((raw_view >> 8) & 0xFF)
            values[view] = (raw_view, f64(gamma), f64(gamma + 8))
            node = u64(node)
        with (out_dir / "gamma_models.txt").open("w", encoding="utf-8") as output:
            output.write("# view gain exponent\n")
            for view in sorted(values):
                _, gain, exponent = values[view]
                output.write("%d %.17g %.17g\n" % (view, gain, exponent))
        with (out_dir / "gamma_models_raw.tsv").open("w", encoding="utf-8") as output:
            output.write("view\traw_view\tgain\texponent\n")
            for view in sorted(values):
                raw_view, gain, exponent = values[view]
                output.write("%d\t%d\t%.17g\t%.17g\n" %
                             (view, raw_view, gain, exponent))
        report.write("GAMMA_MODELS count=%d\n" % len(values))
        if summary_pointer:
            report.write(
                "SUMMARY termination=%d initial_cost=%.17g final_cost=%.17g "
                "successful_steps=%d unsuccessful_steps=%d\n"
                % (i32(summary_pointer + 4), f64(summary_pointer + 40),
                   f64(summary_pointer + 48), i32(summary_pointer + 88),
                   i32(summary_pointer + 92))
            )
            (out_dir / "solver_summary_512.bin").write_bytes(read(summary_pointer, 512))
            iteration_begin = u64(summary_pointer + 64)
            iteration_end = u64(summary_pointer + 72)
            stride = 120
            if iteration_end < iteration_begin or (iteration_end - iteration_begin) % stride:
                raise RuntimeError(
                    "unexpected IterationSummary vector: begin=0x%x end=0x%x"
                    % (iteration_begin, iteration_end)
                )
            iteration_raw = read(iteration_begin, iteration_end - iteration_begin)
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
        report.flush()
        self.enabled = False
        return True


ExposureBreakpoint("*0x55555573ea70", internal=True)
AddResidualBreakpoint("*0x555555a70680", True)
AddResidualBreakpoint("*0x555555a706a0", False)
DynamicRangeBreakpoint("*0x555555743da0", internal=True)
SceneRangeBreakpoint("*0x555555744580", internal=True)
# Ceres 2.2 public Solve wrapper.  The installed binary's bytes at static
# 0x52a890 match ceres-2.2.0 solver.cc.o's public Solve wrapper; the ABI is
# rdi=Options, rsi=Problem, rdx=Summary.
SolveBreakpoint("*0x555555a7e890", internal=True)
ModelBreakpoint("*0x55555573936b", internal=True)
end

run
quit
