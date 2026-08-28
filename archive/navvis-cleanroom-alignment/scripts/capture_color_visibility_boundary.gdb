set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on

starti

python
import gdb
import struct

VISIBILITY_ENTRY = 0x55555572F630
LINEAR_RETURN = 0x55555572F6D1
VISIBILITY_RESULT = 0x55555572F6EF


def reg_float(name):
    return float(gdb.parse_and_eval("$%s.v4_float[0]" % name))


def reg_int(name):
    return int(gdb.parse_and_eval("$" + name))


def f32(address):
    data = gdb.selected_inferior().read_memory(address, 4)
    return struct.unpack("<f", data)[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def u64(address):
    data = gdb.selected_inferior().read_memory(address, 8)
    return struct.unpack("<Q", data)[0]


def set_thread(breakpoint, thread):
    breakpoint.thread = thread.global_num
    return breakpoint


class VisibilityResultBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % VISIBILITY_RESULT, temporary=True, internal=False)
        set_thread(self, thread)

    def stop(self):
        result = reg_int("eax") & 0xff
        gdb.write("OFFICIAL_VISIBILITY_RESULT visible=%d\n" % result)
        gdb.execute("quit")
        return True


class LinearReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % LINEAR_RETURN, temporary=True, internal=False)
        self.thread_object = thread
        set_thread(self, thread)

    def stop(self):
        stack = reg_int("rsp")
        interpolated = reg_float("xmm0")
        ray_range = f32(stack + 4)
        difference = abs(ray_range - interpolated)
        gdb.write(
            "OFFICIAL_LINEAR range=%.17g/0x%08x linear=%.17g/0x%08x "
            "abs_difference=%.17g/0x%08x\n"
            % (
                ray_range,
                f32_bits(ray_range),
                interpolated,
                f32_bits(interpolated),
                difference,
                f32_bits(difference),
            )
        )
        VisibilityResultBreakpoint(self.thread_object)
        return False


class VisibilityEntryBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % VISIBILITY_ENTRY, internal=False)

    def stop(self):
        ray_range = reg_float("xmm0")
        x = reg_float("xmm1")
        y = reg_float("xmm2")
        if not (4.12 < ray_range < 4.14 and 401.0 < x < 402.0 and 219.0 < y < 221.0):
            return False

        depth_map = reg_int("rdi")
        scale = struct.unpack(
            "<d", gdb.selected_inferior().read_memory(depth_map + 0x68, 8)
        )[0]
        scaled_x = struct.unpack("<f", struct.pack("<f", x * scale))[0]
        scaled_y = struct.unpack("<f", struct.pack("<f", y * scale))[0]
        primary_x = int(scaled_x)
        primary_y = int(scaled_y)
        neighbor_x = primary_x + (1 if scaled_x > primary_x + 0.5 else -1)
        neighbor_y = primary_y + (1 if scaled_y > primary_y + 0.5 else -1)
        data = u64(depth_map + 0x18)
        step_pointer = u64(depth_map + 0x50)
        row_step = u64(step_pointer)

        def depth(row, column):
            return f32(data + row * row_step + column * 4)

        threshold = f32(depth_map + 0x78)
        far_threshold = f32(depth_map + 0x74)
        gdb.write(
            "OFFICIAL_VISIBILITY_INPUT object=0x%x range=%.17g/0x%08x "
            "x=%.17g/0x%08x y=%.17g/0x%08x scale=%.17g "
            "threshold=%.17g/0x%08x far=%.17g "
            "primary=(%d,%d) neighbor=(%d,%d) "
            "depth_pp=%.17g/0x%08x depth_np=%.17g/0x%08x "
            "depth_pn=%.17g/0x%08x depth_nn=%.17g/0x%08x\n"
            % (
                depth_map,
                ray_range,
                f32_bits(ray_range),
                x,
                f32_bits(x),
                y,
                f32_bits(y),
                scale,
                threshold,
                f32_bits(threshold),
                far_threshold,
                primary_x,
                primary_y,
                neighbor_x,
                neighbor_y,
                depth(primary_y, primary_x),
                f32_bits(depth(primary_y, primary_x)),
                depth(primary_y, neighbor_x),
                f32_bits(depth(primary_y, neighbor_x)),
                depth(neighbor_y, primary_x),
                f32_bits(depth(neighbor_y, primary_x)),
                depth(neighbor_y, neighbor_x),
                f32_bits(depth(neighbor_y, neighbor_x)),
            )
        )
        self.enabled = False
        thread = gdb.selected_thread()
        gdb.execute("set scheduler-locking on")
        LinearReturnBreakpoint(thread)
        return False


VisibilityEntryBreakpoint()
end

continue
quit
