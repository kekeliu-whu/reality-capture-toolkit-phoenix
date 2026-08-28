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


out_dir = pathlib.Path(os.environ["GAMMA_SAME_STATE_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
vendor_max1_path = pathlib.Path(os.environ["GAMMA_SAME_STATE_VENDOR_MAX1"])
clean_max1_path = pathlib.Path(os.environ["GAMMA_SAME_STATE_CLEAN_MAX1"])
dump_iteration = int(os.environ.get("GAMMA_SAME_STATE_DUMP_ITERATION", "2"))
max_iterations = int(os.environ.get("GAMMA_SAME_STATE_MAX_ITERATIONS", "2"))
if dump_iteration != max_iterations:
    raise gdb.GdbError(
        "this probe uses max_num_iterations as stable vector storage; "
        "dump iteration must equal max iterations"
    )

OPTIONS_SIZE = 488
MAX_ITERATIONS_OFFSET = 104
THREADS_OFFSET = 120
INITIAL_RADIUS_OFFSET = 128
DUMP_ITERATIONS_OFFSET = 376
DUMP_DIRECTORY_OFFSET = 400
DUMP_FORMAT_OFFSET = 432


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def write(address, payload):
    gdb.selected_inferior().write_memory(address, payload)


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def i32(address):
    return struct.unpack("<i", read(address, 4))[0]


def i64(address):
    return struct.unpack("<q", read(address, 8))[0]


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


class SolveFinish(gdb.FinishBreakpoint):
    def __init__(self, frame, options, summary, saved):
        super().__init__(frame, internal=True)
        self.options = options
        self.summary = summary
        self.saved = saved

    def stop(self):
        write(self.options + DUMP_ITERATIONS_OFFSET, self.saved["vector"])
        write(self.options + DUMP_DIRECTORY_OFFSET, self.saved["string"])
        write(self.options + DUMP_FORMAT_OFFSET, self.saved["format"])
        raw = read(self.summary, 512)
        (out_dir / "solver_summary_512.bin").write_bytes(raw)
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
        return False


class SolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        options = int(gdb.parse_and_eval("$rdi"))
        summary = int(gdb.parse_and_eval("$rdx"))
        before = read(options, OPTIONS_SIZE)
        saved = {
            "vector": read(options + DUMP_ITERATIONS_OFFSET, 24),
            "string": read(options + DUMP_DIRECTORY_OFFSET, 32),
            "format": read(options + DUMP_FORMAT_OFFSET, 4),
        }

        write(options + MAX_ITERATIONS_OFFSET, struct.pack("<i", max_iterations))
        write(options + THREADS_OFFSET, struct.pack("<i", 1))
        write(options + INITIAL_RADIUS_OFFSET, struct.pack("<d", 10000.0))

        # Point the one-element dump vector at max_num_iterations.  This probe
        # deliberately dumps the last requested iteration, so the integer is
        # stable until Solve returns.  The original vector/string/format
        # objects are restored before Solver::Options can be destroyed.
        write(options + DUMP_FORMAT_OFFSET, struct.pack("<i", 1))
        iteration_storage = options + MAX_ITERATIONS_OFFSET
        write(
            options + DUMP_ITERATIONS_OFFSET,
            struct.pack("<QQQ", iteration_storage, iteration_storage + 4,
                        iteration_storage + 4),
        )
        string_object = options + DUMP_DIRECTORY_OFFSET
        write(string_object, struct.pack("<Q", string_object + 16))
        write(string_object + 8, struct.pack("<Q", 1))
        write(string_object + 16, b".\0" + b"\0" * 14)

        effective = read(options, OPTIONS_SIZE)
        (out_dir / "solver_options_before.bin").write_bytes(before)
        (out_dir / "solver_options_effective.bin").write_bytes(effective)
        (out_dir / "dump_patch.json").write_text(
            json.dumps({
                "dump_iteration": dump_iteration,
                "max_iterations": max_iterations,
                "num_threads": 1,
                "initial_trust_region_radius": 10000.0,
                "options_address": hex(options),
                "summary_address": hex(summary),
                "options_before_sha256": hashlib.sha256(before).hexdigest(),
                "options_effective_sha256": hashlib.sha256(effective).hexdigest(),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        SolveFinish(gdb.newest_frame(), options, summary, saved)
        self.enabled = False
        return False


class Iter1StateInjectionBreakpoint(gdb.Breakpoint):
    def stop(self):
        minimizer = int(gdb.parse_and_eval("$rdi"))
        iteration = i32(minimizer + 0x190)
        new_evaluation_point = int(gdb.parse_and_eval("$esi"))
        if iteration != 1 or new_evaluation_point != 0:
            return False

        state = u64(minimizer + 0x218)
        state_size = i64(minimizer + 0x220)
        if state_size != 272:
            raise gdb.GdbError("expected 272 state scalars, got %d" % state_size)
        before = read(state, state_size * 8)

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
                    "state block %d is not clean max1: %.17g %.17g"
                    % (block, gain, exponent)
                )
            view = pair_to_view[pair]
            mapped_views.append(view)
            clean_gain, clean_exponent = struct.unpack("<dd", pair)
            vendor_gain, vendor_exponent = vendor_max1[view]
            after[block * 16 : (block + 1) * 16] = struct.pack(
                "<dd", vendor_gain, vendor_exponent
            )
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
                "iteration": iteration,
                "new_evaluation_point": new_evaluation_point,
                "minimizer_address": hex(minimizer),
                "state_address": hex(state),
                "state_scalars": state_size,
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


SolveBreakpoint(
    "ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)",
    internal=True,
)
Iter1StateInjectionBreakpoint(
    "ceres::internal::TrustRegionMinimizer::EvaluateGradientAndJacobian(bool)",
    internal=True,
)
end

run
quit
