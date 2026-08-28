set pagination off
set confirm off
set breakpoint pending on
starti
python
import gdb
import struct


# The allocation address is deterministic for the single-threaded frozen
# probe command.  The allocator reuses it before the final RangeMeasurement
# vector is assembled, so ignore writes until the first endpoint's captured
# origin-x bits appear at the vector head.
ADDRESS = 0x555559435210
EXPECTED_ORIGIN_X_BITS = 0xBBA56F7A


def memory(address, size):
    return bytes(gdb.selected_inferior().read_memory(address, size))


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


class FinalRangeWrite(gdb.Breakpoint):
    def __init__(self):
        super().__init__(
            "*%#x" % ADDRESS,
            type=gdb.BP_WATCHPOINT,
            wp_class=gdb.WP_WRITE,
            internal=True,
        )
        self.hits = 0

    def stop(self):
        self.hits += 1
        bits = struct.unpack("<I", memory(ADDRESS, 4))[0]
        if bits != EXPECTED_ORIGIN_X_BITS:
            return False

        base = image_base("surveyorslam_processing_node")
        pc = int(gdb.parse_and_eval("$pc"))
        gdb.write(
            "MATCHED NODE-0 DESKEWED RANGE WRITE after %d writes; "
            "pc=%#x base_offset=%#x\n" % (self.hits, pc, pc - base)
        )
        gdb.execute("bt 40")
        gdb.execute("info registers rax rbx rcx rdx rsi rdi r8 r9 r10 r11 rsp")
        gdb.execute("x/12wx %#x" % ADDRESS)
        gdb.execute("x/20i $pc-40")
        gdb.execute("quit")
        return True


FinalRangeWrite()
end
continue
