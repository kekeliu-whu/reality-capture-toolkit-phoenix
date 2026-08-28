set pagination off
set confirm off
set print thread-events off
starti
python
import gdb
import json
import os
from pathlib import Path
import struct


OUTPUT = Path(os.environ.get(
    "NAVVIS_PROBE_OUTPUT", "/tmp/navvis_local_icp_caller.json"
))
TARGET_CALL = int(os.environ.get("NAVVIS_PROBE_CALL_INDEX", "10"))


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


BASE = image_base("surveyorslam_processing_node")
calls = 0


class MatchBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*%#x" % (BASE + 0x6C4660), internal=True)

    def stop(self):
        global calls
        call = calls
        calls += 1
        if call != TARGET_CALL:
            return False
        registers = {
            name: int(gdb.parse_and_eval("$" + name))
            for name in ("rdi", "rsi", "rdx", "rcx", "r8", "r9", "rsp")
        }
        initial = bytes(
            gdb.selected_inferior().read_memory(registers["rcx"], 0x40)
        )
        frames = []
        frame = gdb.newest_frame()
        for depth in range(16):
            if frame is None:
                break
            frame_record = {
                "depth": depth,
                "pc_offset": int(frame.pc()) - BASE,
                "name": frame.name(),
            }
            frame_registers = {}
            for name in ("rbx", "rbp", "r12", "r13", "r14", "r15", "rsp"):
                try:
                    frame_registers[name] = int(frame.read_register(name))
                except gdb.error:
                    pass
            frame_record["registers"] = frame_registers
            if depth == 1 and "rbx" in frame_registers:
                work_pose_address = frame_registers["rbx"] + 0x10
                try:
                    work_pose = bytes(
                        gdb.selected_inferior().read_memory(
                            work_pose_address, 0x40
                        )
                    )
                    frame_record["work_pose_address"] = work_pose_address
                    frame_record["work_pose_translation"] = struct.unpack_from(
                        "<3d", work_pose, 0
                    )
                    frame_record["work_pose_quaternion_xyzw"] = struct.unpack_from(
                        "<4d", work_pose, 0x20
                    )
                except gdb.MemoryError:
                    pass
            frames.append(frame_record)
            frame = frame.older()
        record = {
            "call": call,
            "registers": registers,
            "initial_translation": struct.unpack_from("<3d", initial, 0),
            "initial_quaternion_xyzw": struct.unpack_from("<4d", initial, 0x20),
            "frames": frames,
        }
        OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
        gdb.write("captured local ICP caller for call %d\n" % call)
        gdb.execute("quit")
        return False


MatchBreakpoint()
end
continue
