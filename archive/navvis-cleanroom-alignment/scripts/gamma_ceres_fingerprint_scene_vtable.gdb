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


out_dir = pathlib.Path(os.environ["GAMMA_CERES_FINGERPRINT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


class SceneAddResidualBreakpoint(gdb.Breakpoint):
    def stop(self):
        cost = int(gdb.parse_and_eval("$rsi"))
        loss = int(gdb.parse_and_eval("$rdx"))
        parameter_blocks = int(gdb.parse_and_eval("$rcx"))
        parameter_count = int(gdb.parse_and_eval("$r8"))
        vtable = u64(cost)
        functions = [u64(vtable + 8 * index) for index in range(8)]
        (out_dir / "scene_cost_vtable.json").write_text(
            json.dumps({
                "cost_function": cost,
                "loss_function": loss,
                "parameter_blocks": parameter_blocks,
                "parameter_count": parameter_count,
                "vtable": vtable,
                "vtable_functions": functions,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return True


class SceneHelperBreakpoint(gdb.Breakpoint):
    def stop(self):
        SceneAddResidualBreakpoint("*0x555555a706a0", internal=True)
        self.enabled = False
        return False


SceneHelperBreakpoint("*0x555555744580", internal=True)
end

run
quit
