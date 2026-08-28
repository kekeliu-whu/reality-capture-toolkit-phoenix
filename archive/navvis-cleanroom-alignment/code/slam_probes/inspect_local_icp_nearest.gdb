set pagination off
set confirm off
set breakpoint pending on
starti
python
import gdb
import json
import struct

OUTPUT = "/tmp/navvis_local_icp_nearest.json"


def read(fmt, address):
    size = struct.calcsize(fmt)
    data = bytes(gdb.selected_inferior().read_memory(address, size))
    return struct.unpack(fmt, data)


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    if not pid:
        return None
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    return None


class NearestReturn(gdb.FinishBreakpoint):
    def __init__(self, record, index_pointer, distance_pointer):
        super().__init__(gdb.newest_frame(), internal=True)
        self.record = record
        self.index_pointer = index_pointer
        self.distance_pointer = distance_pointer

    def stop(self):
        executable_base = image_base("surveyorslam_processing_node")
        self.record["result"] = bool(int(gdb.parse_and_eval("$al")))
        self.record["index"] = read("<Q", self.index_pointer)[0]
        self.record["squared_distance"] = read("<f", self.distance_pointer)[0]
        self.record["executable_base"] = executable_base
        if executable_base is not None:
            self.record["caller_offset"] = self.record["caller_pc"] - executable_base
        with open(OUTPUT, "w") as stream:
            json.dump(self.record, stream, indent=2, sort_keys=True)
        gdb.write("captured first OctreeUniBN nearest-neighbour query\n")
        gdb.execute("quit")
        return False


class NearestBreakpoint(gdb.Breakpoint):
    def stop(self):
        query_pointer = int(gdb.parse_and_eval("$rsi"))
        index_pointer = int(gdb.parse_and_eval("$rdx"))
        distance_pointer = int(gdb.parse_and_eval("$rcx"))
        caller = gdb.newest_frame().older()
        record = {
            "query": read("<3f", query_pointer),
            "maximum_distance": float(gdb.parse_and_eval("$xmm0.v4_float[0]")),
            "caller_pc": int(caller.pc()),
            "caller_name": caller.name(),
        }
        NearestReturn(record, index_pointer, distance_pointer)
        return False


NearestBreakpoint(
    "_ZNK6navvis10pointcloud11OctreeUniBNIfE19getNearestNeighbourERKN5Eigen6MatrixIfLi3ELi1ELi0ELi3ELi1EEEfPmPf"
)
end
continue
