set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_black_filter_20260827/official_tie_crop/adapter_sizes.log
set logging overwrite on
set logging redirect on
set logging enabled on

starti

python
import gdb
import struct

AFTER_ADAPTER_SIZE = 0x5555557257DE


def register(name):
    return int(gdb.parse_and_eval("$" + name))


def u64(address):
    return struct.unpack("<Q", gdb.selected_inferior().read_memory(address, 8))[0]


class AdapterSizeBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % AFTER_ADAPTER_SIZE, internal=False)
        self.ordinal = 0

    def stop(self):
        adapter = register("r13")
        output = register("rbx")
        vtable = u64(adapter)
        output_begin = u64(output)
        output_end = u64(output + 8)
        output_capacity = u64(output + 16)
        gdb.write(
            "OVS_ADAPTER_SIZE ordinal=%d count=%d adapter=0x%x vtable=0x%x "
            "output=0x%x output_begin=0x%x output_end=0x%x output_capacity=0x%x\n"
            % (self.ordinal, register("rax"), adapter, vtable, output,
               output_begin, output_end, output_capacity)
        )
        gdb.write("OVS_ADAPTER_WORDS ordinal=%d" % self.ordinal)
        for offset in range(0, 0x80, 8):
            try:
                value = u64(adapter + offset)
            except gdb.MemoryError:
                value = 0
            gdb.write(" +0x%x=0x%x" % (offset, value))
        gdb.write("\n")
        self.ordinal += 1
        return False


AdapterSizeBreakpoint()
end

continue
