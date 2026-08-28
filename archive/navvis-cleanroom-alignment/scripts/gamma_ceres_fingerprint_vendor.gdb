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


out_dir = pathlib.Path(os.environ["GAMMA_CERES_FINGERPRINT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
input_path = pathlib.Path(os.environ["GAMMA_CERES_FINGERPRINT_OVS"])
summary_address = 0


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u32(address):
    return struct.unpack("<I", read(address, 4))[0]


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def f64(address):
    return struct.unpack("<d", read(address, 8))[0]


def capture_summary(address):
    raw = read(address, 512)
    (out_dir / "solver_summary_512.bin").write_bytes(raw)
    begin, end, capacity = struct.unpack_from("<QQQ", raw, 64)
    stride = 120
    if end < begin or (end - begin) % stride != 0 or capacity < end:
        raise gdb.GdbError(
            "unexpected iteration vector begin=%#x end=%#x capacity=%#x"
            % (begin, end, capacity)
        )
    iterations_raw = read(begin, end - begin)
    (out_dir / "iterations.bin").write_bytes(iterations_raw)
    iterations = []
    for offset in range(0, len(iterations_raw), stride):
        item = iterations_raw[offset : offset + stride]
        iterations.append({
            "iteration": struct.unpack_from("<i", item, 0)[0],
            "step_is_valid": bool(item[4]),
            "step_is_nonmonotonic": bool(item[5]),
            "step_is_successful": bool(item[6]),
            "cost": struct.unpack_from("<d", item, 8)[0],
            "cost_change": struct.unpack_from("<d", item, 16)[0],
            "gradient_max_norm": struct.unpack_from("<d", item, 24)[0],
            "gradient_norm": struct.unpack_from("<d", item, 32)[0],
            "step_norm": struct.unpack_from("<d", item, 40)[0],
            "relative_decrease": struct.unpack_from("<d", item, 48)[0],
            "trust_region_radius": struct.unpack_from("<d", item, 56)[0],
            "eta": struct.unpack_from("<d", item, 64)[0],
            "linear_solver_iterations": struct.unpack_from("<i", item, 92)[0],
        })
    (out_dir / "iterations.json").write_text(
        json.dumps(iterations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "ceres_version_fingerprint": "2.1.x",
        "termination": struct.unpack_from("<i", raw, 4)[0],
        "initial_cost": struct.unpack_from("<d", raw, 40)[0],
        "final_cost": struct.unpack_from("<d", raw, 48)[0],
        "successful_steps": struct.unpack_from("<i", raw, 88)[0],
        "unsuccessful_steps": struct.unpack_from("<i", raw, 92)[0],
    }
    (out_dir / "vendor_solver_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class ExposureBreakpoint(gdb.Breakpoint):
    def stop(self):
        vector = int(gdb.parse_and_eval("$rsi"))
        begin = u64(vector)
        end = u64(vector + 8)
        original = read(begin, end - begin)
        effective = input_path.read_bytes()
        if len(effective) != len(original):
            raise gdb.GdbError(
                "fixed OVS size mismatch %d != %d" % (len(effective), len(original))
            )
        gdb.selected_inferior().write_memory(begin, effective)
        (out_dir / "effective_exposure_ovs.bin").write_bytes(effective)
        (out_dir / "ovs_capture.json").write_text(
            json.dumps({
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "effective_sha256": hashlib.sha256(effective).hexdigest(),
                "records": len(effective) // 40,
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
        raw = read(options, 488)
        (out_dir / "solver_options_before.bin").write_bytes(raw)
        gdb.selected_inferior().write_memory(options + 120, struct.pack("<i", 1))
        gdb.selected_inferior().write_memory(options + 364, struct.pack("<i", 1))
        gdb.selected_inferior().write_memory(options + 368, b"\x01")
        (out_dir / "solver_options_effective.bin").write_bytes(read(options, 488))
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


ExposureBreakpoint("*0x55555573ea70", internal=True)
SolveBreakpoint("*0x555555a7e890", internal=True)
ModelBreakpoint("*0x55555573936b", internal=True)
end

run
quit
