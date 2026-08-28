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

# Addresses are for nv_colorcloud build-id
# a7586f518009434f5e97891f897aea42675f26a0.  The script stops the official
# single-thread process during final point-wise OVS; it never patches code,
# data, parameters, or the input cloud.
OVS_VIEW_WORKER_ENTRY = 0x555555725F80
VISIBILITY_CALL = 0x5555557264FF
VISIBILITY_NEAREST_RETURN = 0x55555572F687
VISIBILITY_LINEAR_RETURN = 0x55555572F6D1
VISIBILITY_RESULT = 0x55555572F6EF
FINAL_ADAPTER_VTABLE = 0x555555ED0330

target_spec = os.environ.get("PCT_BOUNDARY_TARGETS", "33:426170,90:986541")
TARGET_POINT_BY_VIEW = {
    int(pair.split(":", 1)[0]): int(pair.split(":", 1)[1])
    for pair in target_spec.split(",")
}
OUTPUT = os.environ.get(
    "PCT_BOUNDARY_OUTPUT",
    "work/color_alignment/pct_boundary_vendor_visibility_20260827/vendor_visibility.log",
)

captured = set()
active = {}


def register(name):
    return int(gdb.parse_and_eval("$" + name))


def reg_f32(name):
    bits = int(gdb.parse_and_eval("$%s.v4_int32[0]" % name)) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", bits))[0], bits


def read_u64(address):
    return struct.unpack(
        "<Q", bytes(gdb.selected_inferior().read_memory(address, 8))
    )[0]


def read_f32(address):
    bits = struct.unpack(
        "<I", bytes(gdb.selected_inferior().read_memory(address, 4))
    )[0]
    return struct.unpack("<f", struct.pack("<I", bits))[0], bits


def append(line):
    with open(OUTPUT, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    gdb.write(line + "\n")


def bind_thread(breakpoint, thread):
    breakpoint.thread = thread.global_num
    return breakpoint


class ResultBreakpoint(gdb.Breakpoint):
    def __init__(self, thread, view, point):
        super().__init__("*0x%x" % VISIBILITY_RESULT, internal=False)
        self.view = view
        self.point = point
        bind_thread(self, thread)

    def stop(self):
        self.enabled = False
        result = register("eax") & 0xFF
        state = active.pop((self.view, self.point))
        state["visible"] = result
        append(
            "PCT_BOUNDARY_RESULT view=%d point=%d visible=%d nearest=%.17g/0x%08x "
            "linear=%.17g/0x%08x"
            % (
                self.view,
                self.point,
                result,
                state.get("nearest", float("nan")),
                state.get("nearest_bits", 0),
                state.get("linear", float("nan")),
                state.get("linear_bits", 0),
            )
        )
        captured.add((self.view, self.point))
        gdb.execute("set scheduler-locking off")
        if len(captured) == len(TARGET_POINT_BY_VIEW):
            append("PCT_BOUNDARY_COMPLETE captures=%d" % len(captured))
            gdb.execute("quit")
            return True
        return False


class LinearReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, thread, view, point):
        super().__init__("*0x%x" % VISIBILITY_LINEAR_RETURN, internal=False)
        self.view = view
        self.point = point
        bind_thread(self, thread)

    def stop(self):
        self.enabled = False
        value, bits = reg_f32("xmm0")
        state = active[(self.view, self.point)]
        state["linear"] = value
        state["linear_bits"] = bits
        ResultBreakpoint(gdb.selected_thread(), self.view, self.point)
        return False


class NearestReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, thread, view, point):
        super().__init__("*0x%x" % VISIBILITY_NEAREST_RETURN, internal=False)
        self.view = view
        self.point = point
        bind_thread(self, thread)

    def stop(self):
        self.enabled = False
        value, bits = reg_f32("xmm0")
        state = active[(self.view, self.point)]
        state["nearest"] = value
        state["nearest_bits"] = bits
        LinearReturnBreakpoint(gdb.selected_thread(), self.view, self.point)
        return False


class VisibilityCallBreakpoint(gdb.Breakpoint):
    def __init__(self, thread, view, point):
        super().__init__("*0x%x" % VISIBILITY_CALL, internal=False)
        self.view = view
        self.point = point
        bind_thread(self, thread)

    def stop(self):
        self.enabled = False
        depth_map = register("rdi")
        ray_range, range_bits = reg_f32("xmm0")
        image_x, x_bits = reg_f32("xmm1")
        image_y, y_bits = reg_f32("xmm2")
        scale = struct.unpack(
            "<d", bytes(gdb.selected_inferior().read_memory(depth_map + 0x68, 8))
        )[0]
        scale_f32 = struct.unpack("<f", struct.pack("<f", scale))[0]
        x = struct.unpack("<f", struct.pack("<f", image_x * scale_f32))[0]
        y = struct.unpack("<f", struct.pack("<f", image_y * scale_f32))[0]
        primary_x = int(x // 1.0)
        primary_y = int(y // 1.0)
        neighbor_x = primary_x + (1 if x > primary_x + 0.5 else -1)
        neighbor_y = primary_y + (1 if y > primary_y + 0.5 else -1)
        data = read_u64(depth_map + 0x18)
        step_pointer = read_u64(depth_map + 0x50)
        row_step = read_u64(step_pointer)

        def depth(row, column):
            return read_f32(data + row * row_step + column * 4)

        pp = depth(primary_y, primary_x)
        np = depth(primary_y, neighbor_x)
        pn = depth(neighbor_y, primary_x)
        nn = depth(neighbor_y, neighbor_x)
        far = read_f32(depth_map + 0x74)
        threshold = read_f32(depth_map + 0x78)
        active[(self.view, self.point)] = {}
        append(
            "PCT_BOUNDARY_INPUT view=%d point=%d range=%.17g/0x%08x "
            "image_x=%.17g/0x%08x image_y=%.17g/0x%08x scale=%.17g "
            "scaled_x=%.17g scaled_y=%.17g primary=(%d,%d) neighbor=(%d,%d) "
            "depth_pp=%.17g/0x%08x depth_np=%.17g/0x%08x "
            "depth_pn=%.17g/0x%08x depth_nn=%.17g/0x%08x "
            "far=%.17g/0x%08x threshold=%.17g/0x%08x"
            % (
                self.view,
                self.point,
                ray_range,
                range_bits,
                image_x,
                x_bits,
                image_y,
                y_bits,
                scale,
                x,
                y,
                primary_x,
                primary_y,
                neighbor_x,
                neighbor_y,
                pp[0],
                pp[1],
                np[0],
                np[1],
                pn[0],
                pn[1],
                nn[0],
                nn[1],
                far[0],
                far[1],
                threshold[0],
                threshold[1],
            )
        )
        NearestReturnBreakpoint(gdb.selected_thread(), self.view, self.point)
        return False


class PointIndexWatchpoint(gdb.Breakpoint):
    def __init__(self, thread, address, view, point):
        super().__init__(
            "*(int*)0x%x" % address,
            type=gdb.BP_WATCHPOINT,
            wp_class=gdb.WP_READ,
            internal=False,
        )
        self.view = view
        self.point = point
        bind_thread(self, thread)

    def stop(self):
        self.enabled = False
        if register("r12") != self.point:
            raise gdb.GdbError(
                "target list watchpoint loaded %d, expected %d"
                % (register("r12"), self.point)
            )
        gdb.execute("set scheduler-locking on")
        VisibilityCallBreakpoint(gdb.selected_thread(), self.view, self.point)
        return False


class ViewWorkerBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % OVS_VIEW_WORKER_ENTRY, internal=False)

    def stop(self):
        ovs = register("rdi")
        adapter = read_u64(ovs + 0x90)
        # The same worker body is entered while the PCT renderer is being
        # initialized, where the final-color adapter slot is still null.
        if adapter == 0:
            return False
        try:
            adapter_vtable = read_u64(adapter)
        except gdb.MemoryError:
            return False
        if adapter_vtable != FINAL_ADAPTER_VTABLE:
            return False

        view_record = register("rsi")
        payload = bytes(gdb.selected_inferior().read_memory(view_record, 4))
        camera = payload[1]
        capture = struct.unpack("<H", payload[2:4])[0]
        view = capture * 4 + camera
        point = TARGET_POINT_BY_VIEW.get(view)
        if point is None or (view, point) in captured:
            return False

        candidate_vector = register("r9")
        begin = read_u64(candidate_vector)
        end = read_u64(candidate_vector + 8)
        count = (end - begin) // 4
        raw = bytes(gdb.selected_inferior().read_memory(begin, end - begin))
        values = struct.unpack("<%di" % count, raw)
        try:
            ordinal = values.index(point)
        except ValueError:
            append(
                "PCT_BOUNDARY_NOT_CANDIDATE view=%d point=%d candidates=%d"
                % (view, point, count)
            )
            return False

        address = begin + ordinal * 4
        append(
            "PCT_BOUNDARY_ARM view=%d point=%d candidates=%d ordinal=%d address=0x%x"
            % (view, point, count, ordinal, address)
        )
        if count == 1 and ordinal == 0:
            # The minimal two-record probe gives each target view one candidate;
            # avoid a GDB 15.x Python hardware-watchpoint crash in this case.
            gdb.execute("set scheduler-locking on")
            VisibilityCallBreakpoint(gdb.selected_thread(), view, point)
        else:
            PointIndexWatchpoint(gdb.selected_thread(), address, view, point)
        return False


os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as stream:
    stream.write(
        "PCT_BOUNDARY_VENDOR build_id=a7586f518009434f5e97891f897aea42675f26a0 "
        "targets=%s\n" % target_spec
    )
ViewWorkerBreakpoint()
end

continue
