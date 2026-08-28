set pagination off
set confirm off
starti
python
import gdb
import struct


ADVANCE_OFFSET = 0x6E0F30
RAW_VPTR_OFFSET = 0x1830660
FIRST_RAY_TIME_NS = 1784626878164344000


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


base = image_base("surveyorslam_processing_node")


class FirstDeskewAdvance(gdb.Breakpoint):
    def stop(self):
        owner = int(gdb.parse_and_eval("$rdi"))
        target = int(gdb.parse_and_eval("$rsi")) & 0xFFFFFFFFFFFFFFFF
        vptr = struct.unpack(
            "<Q", bytes(gdb.selected_inferior().read_memory(owner, 8))
        )[0]
        if vptr != base + RAW_VPTR_OFFSET or target != FIRST_RAY_TIME_NS:
            return False
        gdb.write("FIRST PER-RAY RAW IMU ADVANCE\n")
        gdb.execute("bt 50")
        gdb.execute("info registers rax rbx rcx rdx rsi rdi r8 r9 r10 r11 rsp")
        gdb.execute("x/24i $pc")
        gdb.execute("quit")
        return True


FirstDeskewAdvance("*%#x" % (base + ADVANCE_OFFSET), internal=True)
end
continue
