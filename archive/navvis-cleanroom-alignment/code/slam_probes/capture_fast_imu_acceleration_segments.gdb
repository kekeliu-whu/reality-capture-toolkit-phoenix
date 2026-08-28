set pagination off
set confirm off
set print elements 0
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_acceleration_segment_probe/segments.log
set logging overwrite on
set logging enabled on
starti

python
import gdb
import struct


IMAGE_BASE = 0x555555554000
INTEGRATOR = IMAGE_BASE + 0x2586B0
MIDPOINT_RETURN = IMAGE_BASE + 0x237CAB
ACCEL_PRE = IMAGE_BASE + 0x2594C3
ACCEL_POST = IMAGE_BASE + 0x2594C8
MAX_SEGMENTS = 24


def read_u64(address):
    return int(gdb.parse_and_eval(f"*(unsigned long long*){address:#x}"))


def read_f64(address):
    bits = read_u64(address)
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def read_vec(address, count):
    return [read_f64(address + 8 * index) for index in range(count)]


def fmt(values):
    return "[" + ", ".join(f"{value:.17g}" for value in values) + "]"


class State:
    active = False
    frame_rsp = 0
    segment = 0


class IntegratorEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{INTEGRATOR:#x}", internal=False)

    def stop(self):
        stack = int(gdb.parse_and_eval("$rsp"))
        return_address = read_u64(stack)
        if return_address != MIDPOINT_RETURN:
            return False
        State.active = True
        # Six saved registers plus the 0x2b8-byte local frame.
        State.frame_rsp = stack - 0x2E8
        State.segment = 0
        gdb.write(
            "INTEGRATOR_BEGIN "
            f"source_ns={int(gdb.parse_and_eval('$rdx'))} "
            f"target_ns={int(gdb.parse_and_eval('$rcx'))}\n"
        )
        return False


class AccelerationPre(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{ACCEL_PRE:#x}", internal=False)

    def stop(self):
        if not State.active or State.segment >= MAX_SEGMENTS:
            return False
        stack = int(gdb.parse_and_eval("$rsp"))
        if stack != State.frame_rsp:
            return False
        index = State.segment
        r12 = int(gdb.parse_and_eval("$r12"))
        rbp = int(gdb.parse_and_eval("$rbp"))
        gdb.write(
            f"SEGMENT_PRE index={index} "
            f"full_dt={read_f64(stack + 0x158):.17g} "
            f"left_clip={read_f64(stack + 0x160):.17g} "
            f"right_clip={read_f64(stack + 0x168):.17g} "
            f"endpoint0={fmt(read_vec(stack + 0x220, 3))} "
            f"endpoint1={fmt(read_vec(stack + 0x240, 3))} "
            f"state_dv={fmt(read_vec(r12, 3))} "
            f"state_dp={fmt(read_vec(r12 + 0x18, 3))} "
            f"state_q_xyzw={fmt(read_vec(r12 + 0x30, 4))} "
            f"rbp={rbp:#x}\n"
        )
        return False


class AccelerationPost(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{ACCEL_POST:#x}", internal=False)

    def stop(self):
        if not State.active or State.segment >= MAX_SEGMENTS:
            return False
        stack = int(gdb.parse_and_eval("$rsp"))
        if stack != State.frame_rsp:
            return False
        gdb.write(
            f"SEGMENT_POST index={State.segment} "
            f"integral={fmt(read_vec(stack + 0x260, 3))}\n"
        )
        State.segment += 1
        return False


class IntegratorReturn(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{MIDPOINT_RETURN:#x}", internal=False)

    def stop(self):
        if not State.active:
            return False
        gdb.write(f"INTEGRATOR_END captured={State.segment}\n")
        gdb.execute("set logging enabled off")
        gdb.execute("quit")
        return False


IntegratorEntry()
AccelerationPre()
AccelerationPost()
IntegratorReturn()
end

continue
