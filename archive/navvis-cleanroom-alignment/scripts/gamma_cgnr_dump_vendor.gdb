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
input_path = pathlib.Path(os.environ["GAMMA_AUTO_EXPOSURE_INPUT"])
dump_iteration = int(os.environ.get("GAMMA_DUMP_ITERATION", "1"))
max_iterations = int(os.environ.get("GAMMA_MAX_ITERATIONS", str(dump_iteration)))
initial_radius = float(os.environ.get("GAMMA_INITIAL_RADIUS", "10000"))

VECTOR_OFFSET = 376
STRING_OFFSET = 400
DUMP_FORMAT_OFFSET = 432


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def write(address, data):
    gdb.selected_inferior().write_memory(address, data)


class ExposureBreakpoint(gdb.Breakpoint):
    def stop(self):
        vector = int(gdb.parse_and_eval("$rsi"))
        begin, end = struct.unpack("<QQ", read(vector, 16))
        original = read(begin, end - begin)
        effective = input_path.read_bytes()
        if len(original) != len(effective):
            raise gdb.GdbError(
                "fixed exposure OVS size mismatch: %d != %d"
                % (len(original), len(effective))
            )
        write(begin, effective)
        (out_dir / "effective_exposure_ovs.bin").write_bytes(effective)
        (out_dir / "ovs_patch.json").write_text(
            json.dumps(
                {
                    "records": len(effective) // 40,
                    "bytes": len(effective),
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "effective_sha256": hashlib.sha256(effective).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return False


class SolveFinish(gdb.FinishBreakpoint):
    def __init__(self, frame, options, summary, saved_vector, saved_string, saved_format):
        super().__init__(frame, internal=True)
        self.options = options
        self.summary = summary
        self.saved_vector = saved_vector
        self.saved_string = saved_string
        self.saved_format = saved_format

    def stop(self):
        write(self.options + VECTOR_OFFSET, self.saved_vector)
        write(self.options + STRING_OFFSET, self.saved_string)
        write(self.options + DUMP_FORMAT_OFFSET, self.saved_format)

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
        saved_vector = read(options + VECTOR_OFFSET, 24)
        saved_string = read(options + STRING_OFFSET, 32)
        saved_format = read(options + DUMP_FORMAT_OFFSET, 4)

        write(options + 104, struct.pack("<i", max_iterations))
        write(options + 120, struct.pack("<i", 1))
        write(options + 128, struct.pack("<d", initial_radius))
        iteration_storage = options + 104
        write(
            options + VECTOR_OFFSET,
            struct.pack("<QQQ", iteration_storage, iteration_storage + 4,
                        iteration_storage + 4),
        )
        string_object = options + STRING_OFFSET
        write(string_object, struct.pack("<Q", string_object + 16))
        write(string_object + 8, struct.pack("<Q", 1))
        write(string_object + 16, b".\0" + b"\0" * 14)
        write(options + DUMP_FORMAT_OFFSET, struct.pack("<i", 1))

        metadata = {
            "dump_iteration": dump_iteration,
            "max_iterations": max_iterations,
            "num_threads": 1,
            "initial_trust_region_radius": initial_radius,
            "options_address": hex(options),
            "summary_address": hex(summary),
        }
        (out_dir / "dump_patch.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        SolveFinish(
            gdb.newest_frame(), options, summary, saved_vector, saved_string, saved_format
        )
        self.enabled = False
        return False


class StopAfterModelsBreakpoint(gdb.Breakpoint):
    def stop(self):
        # Stop after exposure models have been committed, before color
        # extraction and final PLY writing.  SolveFinish has already restored
        # the temporary Options ownership fields at this point.
        return True


# Hash-frozen nv_colorcloud Build ID a7586f518009434f5e97891f897aea42675f26a0.
ExposureBreakpoint("*0x55555573ea70", internal=True)
SolveBreakpoint("*0x555555a7e890", internal=True)
StopAfterModelsBreakpoint("*0x55555573936b", internal=True)
end

run
quit
