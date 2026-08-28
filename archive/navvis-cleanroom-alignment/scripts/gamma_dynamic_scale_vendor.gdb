set pagination off
set confirm off
set breakpoint pending on
set disable-randomization on
set print thread-events off

python
import json
import os
import pathlib
import struct
import gdb

out_dir = pathlib.Path(os.environ["GAMMA_DYNAMIC_SCALE_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
input_path = pathlib.Path(os.environ["GAMMA_AUTO_EXPOSURE_INPUT"])
raw_weights = None


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


def f64(address):
    return struct.unpack("<d", read(address, 8))[0]


class ExposureBreakpoint(gdb.Breakpoint):
    def stop(self):
        vector = int(gdb.parse_and_eval("$rsi"))
        begin = u64(vector)
        end = u64(vector + 8)
        effective = input_path.read_bytes()
        if len(effective) != end - begin:
            raise RuntimeError("fixed exposure OVS size mismatch")
        gdb.selected_inferior().write_memory(begin, effective)
        self.enabled = False
        return False


class RawWeightsBreakpoint(gdb.Breakpoint):
    def stop(self):
        global raw_weights
        weights = int(gdb.parse_and_eval("$r14"))
        count = int(gdb.parse_and_eval("$rcx"))
        raw_weights = struct.unpack("<" + "d" * count, read(weights, count * 8))
        (out_dir / "raw_weights.bin").write_bytes(
            struct.pack("<" + "d" * count, *raw_weights)
        )
        self.enabled = False
        return False


class ScaleBreakpoint(gdb.Breakpoint):
    def stop(self):
        stack = int(gdb.parse_and_eval("$rsp"))
        weights = int(gdb.parse_and_eval("$r14"))
        count = u64(stack + 0x28)
        values = struct.unpack("<" + "d" * count, read(weights, count * 8))
        result = {
            "count": count,
            "stack_00": f64(stack),
            "stack_08": f64(stack + 8),
            "stack_38": f64(stack + 0x38),
            "xmm0_scale": float(gdb.parse_and_eval("$xmm0.v2_double[0]")),
            "raw_weights": raw_weights,
            "normalized_weights": values,
        }
        (out_dir / "dynamic_scale_inputs.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "normalized_weights.bin").write_bytes(
            struct.pack("<" + "d" * count, *values)
        )
        gdb.execute("quit")
        return False


ExposureBreakpoint("*0x55555573ea70", internal=True)
RawWeightsBreakpoint("*0x55555573a200", internal=True)
# The 136-view fixture takes the even-count loop and reaches the second call
# site.  The preceding odd-element call at 0x55555573a765 is not executed.
ScaleBreakpoint("*0x55555573a7dd", internal=True)
end

run
