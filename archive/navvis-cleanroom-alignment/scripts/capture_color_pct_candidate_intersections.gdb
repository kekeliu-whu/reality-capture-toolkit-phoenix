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
RAYCAST_RETURN = 0x555555736951
CANDIDATE_DISC_COMPARE = 0x555555736544

target_view = int(os.environ["PCT_TARGET_VIEW"])
target_row = int(os.environ["PCT_TARGET_ROW"])
target_column = int(os.environ["PCT_TARGET_COLUMN"])
first_element = None


def register(name):
    return int(gdb.parse_and_eval("$" + name))


def f32_register(name):
    bits = int(gdb.parse_and_eval("$%s.v4_int32[0]" % name)) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", bits))[0], bits


def set_thread(breakpoint, thread):
    breakpoint.thread = thread.global_num
    return breakpoint


class CandidateDiscBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % CANDIDATE_DISC_COMPARE, internal=False)
        set_thread(self, thread)

    def stop(self):
        index = int(gdb.parse_and_eval("*(int*)$rdx"))
        center_distance, center_bits = f32_register("xmm1")
        threshold, threshold_bits = f32_register("xmm4")
        denominator, denominator_bits = f32_register("xmm2")
        gdb.write(
            "PCT_CANDIDATE_DISC index=%d center_distance_squared=%.17g/0x%08x "
            "threshold=%.17g/0x%08x denominator=%.17g/0x%08x accepted=%d\n"
            % (
                index,
                center_distance,
                center_bits,
                threshold,
                threshold_bits,
                denominator,
                denominator_bits,
                1 if center_distance <= threshold else 0,
            )
        )
        return False


class RaycastReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, thread, candidate_breakpoint):
        super().__init__("*0x%x" % RAYCAST_RETURN, temporary=True, internal=False)
        self.candidate_breakpoint = candidate_breakpoint
        set_thread(self, thread)

    def stop(self):
        self.candidate_breakpoint.enabled = False
        packed = register("rax") & 0xFFFFFFFFFFFFFFFF
        status = packed & 0xFFFFFFFF
        range_bits = packed >> 32
        ray_range = struct.unpack("<f", struct.pack("<I", range_bits))[0]
        gdb.write(
            "PCT_CANDIDATE_INTERSECTIONS_COMPLETE status=%d range_squared=%.17g/0x%08x\n"
            % (status, ray_range, range_bits)
        )
        return True


class RaycastCallBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % RAYCAST_CALL, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        self.enabled = False
        candidate_breakpoint = CandidateDiscBreakpoint(self.target_thread)
        RaycastReturnBreakpoint(self.target_thread, candidate_breakpoint)
        return False


class ColumnInitializedBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % COLUMN_INITIALIZED, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        self.enabled = False
        gdb.execute("set $rbx = %d" % target_column)
        RaycastCallBreakpoint(self.target_thread)
        return False


class RowInitializedBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % ROW_INITIALIZED, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        self.enabled = False
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
            "PCT_CANDIDATE_INTERSECTIONS_TARGET view=%d row=%d column=%d thread=%d\n"
            % (target_view, target_row, target_column, thread.num)
        )
        RowInitializedBreakpoint(thread)
        return False


RenderEntryBreakpoint()
end

continue
quit
