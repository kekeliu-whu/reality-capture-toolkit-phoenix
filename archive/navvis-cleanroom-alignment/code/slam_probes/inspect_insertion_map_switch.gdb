set pagination off
set confirm off
set print thread-events off
starti
python
import gdb
import struct

IMAGE_BASE = 0x555555554000
OUTPUT = "/tmp/navvis_vendor_insertion_map_switch.csv"

with open(OUTPUT, "w") as stream:
    stream.write("call,map,ray_vector,count,first_endpoint_x,first_endpoint_y,first_endpoint_z\n")


class InsertionMapSwitch(gdb.Breakpoint):
    """Observe the map selected by the binary for each raw-ray batch.

    The function at image offset 0x48f200 launches the OpenMP insertion once
    for every insertion map.  RDI is the selected surfel map and RSI is the
    vector of 24-byte origin/endpoint rays.  Tracking RDI across the first
    overlap reveals the exact update_all_insertion_maps=false behavior without
    changing the estimator.  This deliberately observes the launcher rather
    than the worker at 0x48ed50, which runs once per OpenMP shard.
    """

    def __init__(self):
        super().__init__("*%#x" % (IMAGE_BASE + 0x48F200), internal=True)
        self.calls = 0
        self.maps = []
        self.second_map_call = None

    def stop(self):
        inferior = gdb.selected_inferior()
        insertion_map = int(gdb.parse_and_eval("$rdi"))
        ray_vector = int(gdb.parse_and_eval("$rsi"))
        begin, end = struct.unpack(
            "<QQ", bytes(inferior.read_memory(ray_vector, 16))
        )
        count = (end - begin) // 24
        first = struct.unpack("<6f", bytes(inferior.read_memory(begin, 24)))
        if insertion_map not in self.maps:
            self.maps.append(insertion_map)
            gdb.write(
                "INSERTION_MAP_NEW call=%d ordinal=%d map=%#x count=%d\n"
                % (self.calls, len(self.maps) - 1, insertion_map, count)
            )
            if len(self.maps) == 2:
                self.second_map_call = self.calls
        with open(OUTPUT, "a") as stream:
            stream.write(
                "%d,%#x,%#x,%d,%.9g,%.9g,%.9g\n"
                % (
                    self.calls,
                    insertion_map,
                    ray_vector,
                    count,
                    first[3],
                    first[4],
                    first[5],
                )
            )
        self.calls += 1
        if self.second_map_call is not None and self.calls >= self.second_map_call + 24:
            gdb.write(
                "INSERTION_MAP_CAPTURE_DONE calls=%d maps=%d second_map_call=%d\n"
                % (self.calls, len(self.maps), self.second_map_call)
            )
            gdb.execute("quit")
        if self.calls >= 700:
            gdb.write(
                "INSERTION_MAP_CAPTURE_LIMIT calls=%d maps=%d\n"
                % (self.calls, len(self.maps))
            )
            gdb.execute("quit")
        return False


InsertionMapSwitch()
end
continue
