set pagination off
set confirm off
starti
python
import gdb
import json
from pathlib import Path


OUTPUT = Path("/media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_surfel_update_order.json")
OFFSETS = {
    "insert_worker": 0x48ED50,
    "maintenance_worker_parallel": 0x48FC60,
    "maintenance_worker_serial": 0x4990B0,
    "merge_driver": 0x4987D0,
    "merge": 0x49B530,
    "match": 0x6C4660,
}


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


events = []
match_count = 0


class Event(gdb.Breakpoint):
    def __init__(self, name, address):
        super().__init__("*%#x" % address, internal=True)
        self.name = name

    def stop(self):
        global match_count
        events.append({
            "index": len(events),
            "name": self.name,
            "before_match": match_count,
            "thread": gdb.selected_thread().num,
        })
        if self.name == "match":
            match_count += 1
            if match_count >= 3:
                OUTPUT.write_text(json.dumps(events, indent=2) + "\n")
                gdb.write("captured %d surfel update-order events\n" % len(events))
                gdb.execute("quit")
        return False


base = image_base("surveyorslam_processing_node")
for name, offset in OFFSETS.items():
    Event(name, base + offset)
end
continue
