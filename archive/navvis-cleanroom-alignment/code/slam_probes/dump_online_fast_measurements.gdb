set pagination off
set confirm off
set logging enabled off
set print thread-events off
starti

python
import gdb
import os
import pathlib
import struct


IMAGE_BASE = 0x555555554000
CERES_SOLVE = IMAGE_BASE + 0xEDD070
ACCELERATION_EVALUATE = IMAGE_BASE + 0x5E7D20
ROTATION_EVALUATE = IMAGE_BASE + 0x5E3A80
TARGET_CERES_CALL = int(os.environ.get("NAVVIS_PROBE_CERES_CALL", "17"))
OUTPUT = pathlib.Path(os.environ["NAVVIS_PROBE_MEASUREMENTS"])
EXPECTED_ACCELERATIONS = 2658
EXPECTED_ROTATIONS = 2659


def functor_values(value_count):
    cost = int(gdb.parse_and_eval("$rdi"))
    functor = int(gdb.parse_and_eval(f"*(void**)({cost:#x}+40)"))
    return struct.unpack(
        f"<{value_count}d",
        gdb.selected_inferior().read_memory(functor, 8 * value_count).tobytes(),
    )


class MeasurementBreakpoint(gdb.Breakpoint):
    def __init__(self, address, expected_count, value_count):
        super().__init__(f"*{address:#x}", internal=True)
        self.enabled = False
        self.expected_count = expected_count
        self.value_count = value_count
        self.values = []

    def stop(self):
        self.values.append(functor_values(self.value_count))
        if len(self.values) >= self.expected_count:
            self.enabled = False
        return False


acceleration = MeasurementBreakpoint(
    ACCELERATION_EVALUATE, EXPECTED_ACCELERATIONS, 6
)
rotation = MeasurementBreakpoint(ROTATION_EVALUATE, EXPECTED_ROTATIONS, 5)


class SolveReturn(gdb.FinishBreakpoint):
    def __init__(self):
        super().__init__(gdb.newest_frame(), internal=True)

    def stop(self):
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="ascii") as stream:
            for index, values in enumerate(acceleration.values):
                stream.write(
                    "ACCELERATION_MEASUREMENT "
                    + str(index)
                    + " "
                    + " ".join(format(value, ".17g") for value in values)
                    + "\n"
                )
            for index, values in enumerate(rotation.values):
                stream.write(
                    "DELTA_ROTATION_MEASUREMENT "
                    + str(index)
                    + " "
                    + " ".join(format(value, ".17g") for value in values)
                    + "\n"
                )
        gdb.write(
            f"ONLINE_MEASUREMENTS acceleration={len(acceleration.values)} "
            f"rotation={len(rotation.values)} output={OUTPUT}\n"
        )
        gdb.execute("quit")
        return False


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{CERES_SOLVE:#x}", internal=True)
        self.count = 0

    def stop(self):
        self.count += 1
        if self.count == TARGET_CERES_CALL:
            acceleration.enabled = True
            rotation.enabled = True
            SolveReturn()
            self.enabled = False
        return False


SolveEntry()
end

continue
