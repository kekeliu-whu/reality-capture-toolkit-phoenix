set pagination off
set confirm off
set breakpoint pending on
starti
python
import gdb


WORKER_OFFSET = 0x48ED50


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


class FirstSurfelWorker(gdb.Breakpoint):
    def stop(self):
        gdb.write("\nFIRST SURFEL WORKER STACK\n")
        gdb.execute("bt 30")
        gdb.execute("info registers rdi rsi rdx rcx r8 r9 rsp")
        gdb.execute("quit")
        return True


base = image_base("surveyorslam_processing_node")
FirstSurfelWorker("*%#x" % (base + WORKER_OFFSET), internal=True)
end
continue
