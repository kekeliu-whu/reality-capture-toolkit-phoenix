set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on

starti

python
import gdb
import os
import struct

RENDER_ENTRY = 0x5555557337F0
ROW_INITIALIZED = 0x55555573678F
COLUMN_INITIALIZED = 0x555555736807
RAYCAST_CALL = 0x55555573694C

target_view = int(os.environ["PCT_TARGET_VIEW"])
target_row = int(os.environ["PCT_TARGET_ROW"])
target_column = int(os.environ["PCT_TARGET_COLUMN"])
first_element = None


def register(name):
    return int(gdb.parse_and_eval("$" + name))


def f32_register(name):
    bits = int(gdb.parse_and_eval("$%s.v4_int32[0]" % name)) & 0xFFFFFFFF
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    return value, bits


def emit(name):
    value, bits = f32_register(name)
    return "%s=%.17g/0x%08x" % (name, value, bits)


def set_thread(breakpoint, thread):
    breakpoint.thread = thread.global_num
    return breakpoint


stages = [
    (0x5555557364C5, "denominator", ("xmm2",)),
    (0x5555557364E5, "origin_dot", ("xmm0",)),
    (0x5555557364F1, "point_dot", ("xmm6",)),
    (0x555555736500, "distance", ("xmm0",)),
    (0x555555736520, "intersection", ("xmm5", "xmm6", "xmm0")),
    (0x555555736567, "range_delta", ("xmm0", "xmm4", "xmm3")),
    (0x55555573658C, "range_squared", ("xmm0",)),
]


class ArithmeticBreakpoint(gdb.Breakpoint):
    def __init__(self, thread, stage):
        address, _, _ = stages[stage]
        super().__init__("*0x%x" % address, temporary=True, internal=False)
        self.target_thread = thread
        self.stage = stage
        set_thread(self, thread)

    def stop(self):
        _, label, registers = stages[self.stage]
        gdb.write("PCT_ARITHMETIC %s %s\n" % (label, " ".join(emit(r) for r in registers)))
        if self.stage + 1 < len(stages):
            ArithmeticBreakpoint(self.target_thread, self.stage + 1)
        return False


class RaycastCallBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % RAYCAST_CALL, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        gdb.write("PCT_ARITHMETIC_BEGIN\n")
        ArithmeticBreakpoint(self.target_thread, 0)
        return False


class ColumnInitializedBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % COLUMN_INITIALIZED, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        gdb.execute("set $rbx = %d" % target_column)
        RaycastCallBreakpoint(self.target_thread)
        return False


class RowInitializedBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % ROW_INITIALIZED, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        stack = register("rsp")
        gdb.execute("set *(unsigned long long*)0x%x = %d" % (stack + 0x20, target_row))
        ColumnInitializedBreakpoint(self.target_thread)
        return False


class RenderEntryBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % RENDER_ENTRY, internal=False)

    def stop(self):
        global first_element
        element = register("rdx")
        if first_element is None:
            first_element = element
        view = (element - first_element) // 16
        if view != target_view:
            return False
        thread = gdb.selected_thread()
        self.enabled = False
        gdb.execute("set scheduler-locking on")
        gdb.write(
            "PCT_ARITHMETIC_TARGET view=%d row=%d column=%d thread=%d\n"
            % (target_view, target_row, target_column, thread.num)
        )
        RowInitializedBreakpoint(thread)
        return False


RenderEntryBreakpoint()
end

continue
quit
