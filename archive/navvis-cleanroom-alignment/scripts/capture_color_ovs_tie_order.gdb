set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_black_filter_20260827/official_tie_crop/tie_order.log
set logging overwrite on
set logging redirect on
set logging enabled on

starti

python
import gdb
import struct

SELECT_ENTRY = 0x5555557256F0
CANDIDATE_ENTRY = 0x55555572EEE0
COMMIT_EXIT = 0x555555726777
SELECT_COMPLETE = 0x5555557258D2
TARGET_INDICES = (5931, 13477)
FULL_POINT_COUNT = 2857623
FINAL_ADAPTER_VTABLE = 0x555555ED0330
output_object = 0
candidate_by_thread = {}


def register(name):
    return int(gdb.parse_and_eval("$" + name))


def u64(address):
    return struct.unpack("<Q", gdb.selected_inferior().read_memory(address, 8))[0]


class CandidateBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % CANDIDATE_ENTRY, internal=False)
        self.ordinal = 0

    def stop(self):
        source = register("rsi")
        payload = bytes(gdb.selected_inferior().read_memory(source, 4))
        camera = payload[1]
        capture = struct.unpack("<H", payload[2:4])[0]
        score = float(gdb.parse_and_eval("$xmm0.v4_float[0]"))
        output_begin = u64(output_object)
        retained_record = u64(register("rsp") + 0x50)
        point = ((retained_record - output_begin) // 40
                 if output_begin <= retained_record < output_begin + 80 else -1)
        thread = gdb.selected_thread().global_num
        candidate_by_thread[thread] = (point, capture * 4 + camera, score)
        gdb.write(
            "OVS_CANDIDATE ordinal=%d thread=%d point=%d view=%d capture=%d camera=%d "
            "score=%.17g retained=0x%x output_begin=0x%x\n"
            % (self.ordinal, thread, point, capture * 4 + camera, capture, camera, score,
               retained_record, output_begin)
        )
        self.ordinal += 1
        return False


class CommitBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % COMMIT_EXIT, internal=False)
        self.ordinal = 0

    def stop(self):
        thread = gdb.selected_thread().global_num
        point, view, score = candidate_by_thread.get(thread, (-1, -1, float("nan")))
        record = register("r11")
        output_begin = u64(output_object)
        actual_point = ((record - output_begin) // 40
                        if output_begin <= record < output_begin + 80 else -1)
        payload = bytes(gdb.selected_inferior().read_memory(record, 40))
        slots = []
        for slot in range(5):
            item = payload[slot * 8:(slot + 1) * 8]
            capture = struct.unpack(">H", item[0:2])[0]
            camera = item[2]
            quality = struct.unpack("<H", item[6:8])[0]
            slots.append((capture * 4 + camera, quality))
        gdb.write(
            "OVS_COMMIT ordinal=%d thread=%d point=%d candidate_point=%d "
            "view=%d score=%.17g slots=%s\n"
            % (self.ordinal, thread, actual_point, point, view, score, slots)
        )
        self.ordinal += 1
        return False


class CompleteBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % SELECT_COMPLETE, internal=False)

    def stop(self):
        begin = u64(output_object)
        end = u64(output_object + 8)
        count = (end - begin) // 40
        if count != len(TARGET_INDICES):
            return False
        end = begin + count * 40
        path = "/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_black_filter_20260827/official_tie_crop/tie_order_ovs.bin"
        gdb.execute("dump binary memory %s 0x%x 0x%x" % (path, begin, end))
        gdb.write("OVS_TIE_CAPTURE_COMPLETE begin=0x%x count=%d\n" % (begin, count))
        gdb.execute("quit")
        return True


class SelectEntryBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % SELECT_ENTRY, internal=False)

    def stop(self):
        global output_object
        ovs = register("rdi")
        adapter = u64(ovs + 0x90)
        vtable = u64(adapter)
        if vtable == FINAL_ADAPTER_VTABLE:
            # The final PointXYZRGBINormal adapter owns a packed 14-byte
            # position/normal-code vector directly at +8/+16.  The earlier
            # exposure adapter instead wraps a 48-byte PCL cloud at +8.
            cloud = adapter
            begin_offset = 0x08
            end_offset = 0x10
            record_bytes = 14
        else:
            cloud = u64(adapter + 0x08)
            begin_offset = 0x30
            end_offset = 0x38
            record_bytes = 48
        begin = u64(cloud + begin_offset)
        end = u64(cloud + end_offset)
        count = (end - begin) // record_bytes
        gdb.write(
            "OVS_SELECT_INPUT adapter=0x%x vtable=0x%x cloud=0x%x begin=0x%x "
            "stride=%d count=%d\n"
            % (adapter, vtable, cloud, begin, record_bytes, count)
        )
        if count != FULL_POINT_COUNT:
            return False
        output_object = register("rdx")
        inferior = gdb.selected_inferior()
        records = [bytes(inferior.read_memory(begin + i * record_bytes, record_bytes))
                   for i in TARGET_INDICES]
        for ordinal, record in enumerate(records):
            inferior.write_memory(begin + ordinal * record_bytes, record)
        inferior.write_memory(cloud + end_offset,
                              struct.pack("<Q", begin + len(records) * record_bytes))
        gdb.write(
            "OVS_PATCHED_INPUT adapter=0x%x cloud=0x%x begin=0x%x indices=%s\n"
            % (adapter, cloud, begin, TARGET_INDICES)
        )
        gdb.execute("set scheduler-locking on")
        CandidateBreakpoint()
        CommitBreakpoint()
        CompleteBreakpoint()
        self.enabled = False
        return False


SelectEntryBreakpoint()
end

continue
