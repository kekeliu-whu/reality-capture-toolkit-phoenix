set pagination off
set confirm off
starti

python
import gdb


class MeasurementEntry(gdb.Breakpoint):
    def __init__(self, address, label, count, value_count):
        super().__init__(f"*{address:#x}", internal=False)
        self.label = label
        self.expected_count = count
        self.value_count = value_count
        self.seen = 0

    def stop(self):
        this_pointer = int(gdb.parse_and_eval("$rdi"))
        functor = int(gdb.parse_and_eval(f"*(void**)({this_pointer:#x}+40)"))
        values = [
            float(gdb.parse_and_eval(f"*(double*)({functor:#x}+{8 * index})"))
            for index in range(self.value_count)
        ]
        gdb.write(
            self.label
            + f" {self.seen} "
            + " ".join(format(value, ".17g") for value in values)
            + "\n"
        )
        self.seen += 1
        if self.seen == self.expected_count:
            self.enabled = False
        return False


MeasurementEntry(0x5555557e2250, "ACCELERATION_MEASUREMENT", 2658, 5)
MeasurementEntry(0x5555557ddfb0, "DELTA_ROTATION_MEASUREMENT", 2659, 4)
end

continue
