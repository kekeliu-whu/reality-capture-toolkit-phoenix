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
QUERY_RETURN = 0x5555557363ED
RAYCAST_RETURN = 0x555555736951

target_view = int(os.environ["PCT_TARGET_VIEW"])
target_row = int(os.environ["PCT_TARGET_ROW"])
target_column = int(os.environ["PCT_TARGET_COLUMN"])
first_element = None


def u64(address):
    memory = gdb.selected_inferior().read_memory(address, 8)
    return struct.unpack("<Q", memory)[0]


def i32(address):
    memory = gdb.selected_inferior().read_memory(address, 4)
    return struct.unpack("<i", memory)[0]


def f32(address):
    memory = gdb.selected_inferior().read_memory(address, 4)
    return struct.unpack("<f", memory)[0]


def register(name):
    return int(gdb.parse_and_eval("$" + name))


def set_thread(breakpoint, thread):
    breakpoint.thread = thread.global_num
    return breakpoint


class RaycastReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % RAYCAST_RETURN, temporary=True, internal=False)
        set_thread(self, thread)

    def stop(self):
        packed = register("rax") & 0xFFFFFFFFFFFFFFFF
        status = packed & 0xFFFFFFFF
        range_bits = packed >> 32
        ray_range = struct.unpack("<f", struct.pack("<I", range_bits))[0]
        gdb.write(
            "PCT_RAYCAST_RESULT status=%d range_bits=0x%08x range=%.17g\n"
            % (status, range_bits, ray_range)
        )
        gdb.write("PCT_CANDIDATE_CAPTURE_COMPLETE\n")
        return True


class QueryReturnBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % QUERY_RETURN, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        count = register("eax") & 0xFFFFFFFF
        vector = register("r14")
        begin = u64(vector)
        end = u64(vector + 8)
        size = (end - begin) // 4
        raycaster = register("r12")
        octree = u64(raycaster + 0x10)
        point_cloud_holder = u64(raycaster + 0x18)
        surfel_base = u64(point_cloud_holder)
        stack = register("rsp")
        origin = (f32(stack + 0x40), f32(stack + 0x44), f32(stack + 0x48))
        ray = (f32(stack + 0x0C), f32(stack + 0x08), f32(stack + 0x00))
        maximum_entries = register("ebp") & 0xFFFFFFFF
        previous_count = register("r15d") & 0xFFFFFFFF
        gdb.write(
            "PCT_QUERY_RESULT returned_count=%d vector_size=%d max_entries=%d "
            "previous_count=%d vector=0x%x surfels=0x%x "
            "octree=0x%x bounds=((%.17g %.17g) (%.17g %.17g) (%.17g %.17g)) "
            "origin=(%.17g %.17g %.17g) ray=(%.17g %.17g %.17g)\n"
            % (
                count,
                size,
                maximum_entries,
                previous_count,
                vector,
                surfel_base,
                octree,
                struct.unpack("<d", gdb.selected_inferior().read_memory(octree + 0x68, 8))[0],
                struct.unpack("<d", gdb.selected_inferior().read_memory(octree + 0x70, 8))[0],
                struct.unpack("<d", gdb.selected_inferior().read_memory(octree + 0x78, 8))[0],
                struct.unpack("<d", gdb.selected_inferior().read_memory(octree + 0x80, 8))[0],
                struct.unpack("<d", gdb.selected_inferior().read_memory(octree + 0x88, 8))[0],
                struct.unpack("<d", gdb.selected_inferior().read_memory(octree + 0x90, 8))[0],
                origin[0],
                origin[1],
                origin[2],
                ray[0],
                ray[1],
                ray[2],
            )
        )
        for ordinal in range(size):
            index = i32(begin + ordinal * 4)
            surfel = surfel_base + index * 32
            values = tuple(f32(surfel + component * 4) for component in range(8))
            gdb.write(
                "PCT_CANDIDATE ordinal=%d evaluated=%d index=%d "
                "xyz=(%.17g %.17g %.17g) normal=(%.17g %.17g %.17g) "
                "tail=(%.17g %.17g)\n"
                % (
                    ordinal,
                    1 if ordinal < count else 0,
                    index,
                    values[0],
                    values[1],
                    values[2],
                    values[4],
                    values[5],
                    values[6],
                    values[3],
                    values[7],
                )
            )
        RaycastReturnBreakpoint(self.target_thread)
        return False


class RaycastCallBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % RAYCAST_CALL, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        origin = register("rsi")
        ray = register("rdx")
        gdb.write(
            "PCT_RAYCAST_CALL origin=(%.17g %.17g %.17g) "
            "ray=(%.17g %.17g %.17g)\n"
            % (
                f32(origin),
                f32(origin + 4),
                f32(origin + 8),
                f32(ray),
                f32(ray + 4),
                f32(ray + 8),
            )
        )
        QueryReturnBreakpoint(self.target_thread)
        return False


class ColumnInitializedBreakpoint(gdb.Breakpoint):
    def __init__(self, thread):
        super().__init__("*0x%x" % COLUMN_INITIALIZED, temporary=True, internal=False)
        self.target_thread = thread
        set_thread(self, thread)

    def stop(self):
        gdb.execute("set $rbx = %d" % target_column)
        gdb.write("PCT_SKIP_TO_COLUMN column=%d\n" % target_column)
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
        gdb.write("PCT_SKIP_TO_ROW row=%d\n" % target_row)
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
            gdb.write("PCT_VIEW_ELEMENT_BASE 0x%x\n" % first_element)
        view = (element - first_element) // 16
        if view != target_view:
            return False
        thread = gdb.selected_thread()
        self.enabled = False
        gdb.execute("set scheduler-locking on")
        gdb.write(
            "PCT_TARGET_RENDER view=%d row=%d column=%d thread=%d element=0x%x\n"
            % (target_view, target_row, target_column, thread.num, element)
        )
        RowInitializedBreakpoint(thread)
        return False


RenderEntryBreakpoint()
end

continue
quit
