set pagination off
set confirm off
set print thread-events off
starti
python
import gdb
import json
import os
import struct
from pathlib import Path


OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_stage2_exact_preintegration_steps.json"
))
records = []
active_output = None
residual_address = None


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


def doubles(address, count):
    return struct.unpack(
        "<%dd" % count,
        bytes(gdb.selected_inferior().read_memory(address, 8 * count)),
    )


def state(address):
    return {
        "delta_position": doubles(address, 3),
        "delta_velocity": doubles(address + 0x18, 3),
        "delta_rotation_xyzw": doubles(address + 0x30, 4),
        "raw_output_doubles": doubles(address, 20),
    }


class StepBreakpoint(gdb.Breakpoint):
    def stop(self):
        if active_output is None:
            return False
        stack = int(gdb.parse_and_eval("$rsp"))
        records.append({
            "stage": "interval_complete",
            "full_dt": doubles(stack + 0x158, 1)[0],
            "left_clip": doubles(stack + 0x160, 1)[0],
            "right_clip": doubles(stack + 0x168, 1)[0],
            "active_dt": doubles(stack + 0x08, 1)[0],
            **state(active_output),
        })
        return False


class PreintegrationFinish(gdb.FinishBreakpoint):
    def stop(self):
        global active_output
        records.append({"stage": "preintegration_finish", **state(active_output)})
        active_output = None
        return False


class PreintegrationBreakpoint(gdb.Breakpoint):
    def stop(self):
        global active_output
        active_output = int(gdb.parse_and_eval("$rdi"))
        records.append({
            "stage": "preintegration_entry",
            "with_translation": int(gdb.parse_and_eval("$rsi")) & 0xff,
            "start_ns": int(gdb.parse_and_eval("$rdx")),
            "end_ns": int(gdb.parse_and_eval("$rcx")),
        })
        PreintegrationFinish(gdb.newest_frame(), internal=True)
        self.enabled = False
        return False


class FinalRigidComposeBreakpoint(gdb.Breakpoint):
    def stop(self):
        first = int(gdb.parse_and_eval("$rsi"))
        second = int(gdb.parse_and_eval("$rdx"))
        output = int(gdb.parse_and_eval("$rdi"))
        records.append({
            "stage": "final_rigid_compose_entry",
            "first": doubles(first, 8),
            "second": doubles(second, 8),
            "output_address": output,
        })
        return False


class RotationErrorBreakpoint(gdb.Breakpoint):
    def stop(self):
        quaternion = int(gdb.parse_and_eval("$r12"))
        records.append({
            "stage": "rotation_error_before_log",
            "quaternion_xyzw": doubles(quaternion, 4),
        })
        return False


class ExactFinish(gdb.FinishBreakpoint):
    def stop(self):
        records.append({
            "stage": "exact_residual",
            "residual": doubles(residual_address, 9),
        })
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(records, indent=2) + "\n")
        gdb.write("captured Stage2 Exact preintegration steps\n")
        gdb.execute("quit")
        return False


BASE = image_base("compute_trajectories")
step_breakpoint = StepBreakpoint("*%#x" % (BASE + 0x258b3c), internal=True)


class ExactBreakpoint(gdb.Breakpoint):
    def stop(self):
        global residual_address
        residual_address = int(gdb.parse_and_eval("$rdx"))
        # Force the residual-only path. It evaluates the same templated functor
        # with doubles and avoids obscuring the integration state with Jets.
        gdb.execute("set $rcx = 0")
        PreintegrationBreakpoint("*%#x" % (BASE + 0x2586b0), internal=True)
        FinalRigidComposeBreakpoint("*%#x" % (BASE + 0x25d4bd), internal=True)
        RotationErrorBreakpoint("*%#x" % (BASE + 0x25d63e), internal=True)
        ExactFinish(gdb.newest_frame(), internal=True)
        self.enabled = False
        return False


ExactBreakpoint("*%#x" % (BASE + 0x272770), internal=True)
end
continue
