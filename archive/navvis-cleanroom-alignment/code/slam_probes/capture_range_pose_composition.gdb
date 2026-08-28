set pagination off
set confirm off
set disable-randomization on
starti
python
import gdb
import json
import os
import struct

IMAGE_BASE = 0x555555554000
COMPOSE_CALL = IMAGE_BASE + 0x4A7AFE
COMPOSE_RETURN = IMAGE_BASE + 0x4A7B03

OUTPUT_DIR = os.environ.get(
    "NAVVIS_RANGE_POSE_COMPOSE_DIR", "/tmp/navvis-range-pose-compose"
)
CAPTURE_COUNT = int(os.environ.get("NAVVIS_RANGE_POSE_COMPOSE_COUNT", "4"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {"compose_call_rva": "0x4a7afe", "captures": []}
pending_by_thread = {}


def inferior():
    return gdb.selected_inferior()


def read(address, size):
    return bytes(inferior().read_memory(address, size))


def decode_pose(payload):
    values = struct.unpack("<8d", payload)
    return {
        "translation": list(values[:3]),
        "padding": values[3],
        "quaternion_xyzw": list(values[4:]),
        "hex": payload.hex(),
    }


def save_state():
    path = os.path.join(OUTPUT_DIR, "trace.json")
    with open(path, "w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)


class ReturnBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % COMPOSE_RETURN, internal=True)

    def stop(self):
        thread_number = gdb.selected_thread().num
        pending = pending_by_thread.pop(thread_number, None)
        if pending is None:
            return False
        capture, output_address = pending
        capture["output"] = decode_pose(read(output_address, 64))
        save_state()
        gdb.write(
            "captured range pose composition %d/%d\n"
            % (len(state["captures"]), CAPTURE_COUNT)
        )
        if len(state["captures"]) >= CAPTURE_COUNT:
            gdb.execute("quit")
        return False


class CallBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % COMPOSE_CALL, internal=True)

    def stop(self):
        if len(state["captures"]) >= CAPTURE_COUNT:
            return False
        output_address = int(gdb.parse_and_eval("$rdi"))
        lhs_address = int(gdb.parse_and_eval("$rsi"))
        rhs_address = int(gdb.parse_and_eval("$rdx"))
        stack_pointer = int(gdb.parse_and_eval("$rsp"))
        capture = {
            "index": len(state["captures"]),
            "output_address": output_address,
            "lhs_address": lhs_address,
            "rhs_address": rhs_address,
            "lhs": decode_pose(read(lhs_address, 64)),
            "rhs": decode_pose(read(rhs_address, 64)),
            "pose_at_stack_0x120": decode_pose(read(stack_pointer + 0x120, 64)),
            "pose_at_stack_0x160": decode_pose(read(stack_pointer + 0x160, 64)),
            "output_before_compose": decode_pose(read(output_address, 64)),
            "output": None,
        }
        state["captures"].append(capture)
        pending_by_thread[gdb.selected_thread().num] = (capture, output_address)
        save_state()
        return False


CallBreakpoint()
ReturnBreakpoint()
end
continue
