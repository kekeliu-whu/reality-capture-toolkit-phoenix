set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on

starti

python
import gdb

RENDER_ENTRY = 0x5555557337f0
RENDER_COMPLETE = 0x55555573395b

entries = {}


class RenderEntryBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % RENDER_ENTRY, internal=False)

    def stop(self):
        thread = gdb.selected_thread()
        element = int(gdb.parse_and_eval("$rdx"))
        view = int(gdb.parse_and_eval("*(unsigned long long*)$rdx"))
        entries[element] = view
        gdb.write(
            "PCT_RENDER_ENTRY thread=%d element=0x%x view=0x%x\n"
            % (thread.num, element, view)
        )
        return False


class RenderCompleteBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % RENDER_COMPLETE, internal=False)
        self.count = 0

    def stop(self):
        thread = gdb.selected_thread()
        element = int(gdb.parse_and_eval("$rbp"))
        gdb.write(
            "PCT_RENDER_COMPLETE thread=%d element=0x%x ordinal=%d\n"
            % (thread.num, element, self.count)
        )
        self.count += 1
        if self.count == 136:
            ordered = sorted(entries.items())
            gdb.write("PCT_VIEW_MAPPING_BEGIN\n")
            for index, (element_address, view_address) in enumerate(ordered):
                gdb.write(
                    "PCT_VIEW index=%d element=0x%x view=0x%x\n"
                    % (index, element_address, view_address)
                )
            gdb.write("PCT_VIEW_MAPPING_END\n")
        return False


RenderEntryBreakpoint()
RenderCompleteBreakpoint()
end

continue
quit
