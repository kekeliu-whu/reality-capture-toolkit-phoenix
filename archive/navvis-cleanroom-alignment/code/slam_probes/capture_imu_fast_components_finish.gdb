set pagination off
set confirm off
starti

python
import gdb
import os

component_skip = int(os.environ.get("NAVVIS_IMU_FAST_COMPONENT_SKIP", "0"))


class ResidualFinish(gdb.FinishBreakpoint):
    def __init__(self, frame, label, residual_pointer, residual_count):
        super().__init__(frame, internal=True)
        self.label = label
        self.residual_pointer = residual_pointer
        self.residual_count = residual_count

    def stop(self):
        gdb.write(f"{self.label}_RESIDUALS\n")
        gdb.execute(f"x/{self.residual_count}gf {self.residual_pointer:#x}")
        return False


class ComponentEntry(gdb.Breakpoint):
    def __init__(self, address, label, parameter_labels, parameter_sizes):
        super().__init__(f"*{address:#x}", internal=False)
        self.label = label
        self.parameter_labels = parameter_labels
        self.parameter_sizes = parameter_sizes
        self.seen = 0

    def stop(self):
        self.seen += 1
        if self.seen <= component_skip:
            return False
        self.enabled = False
        this_pointer = int(gdb.parse_and_eval("$rdi"))
        parameters = int(gdb.parse_and_eval("$rsi"))
        residuals = int(gdb.parse_and_eval("$rdx"))
        residual_count = int(gdb.parse_and_eval(f"*(int*)({this_pointer:#x}+32)"))
        functor = int(gdb.parse_and_eval(f"*(void**)({this_pointer:#x}+40)"))
        gdb.write(
            f"{self.label}_ENTRY this={this_pointer:#x} residuals={residuals:#x} "
            f"residual_count={residual_count} functor={functor:#x}\n"
        )
        gdb.write(f"{self.label}_FUNCTOR_DOUBLES\n")
        gdb.execute(f"x/12gf {functor:#x}")
        for index, (name, size) in enumerate(
            zip(self.parameter_labels, self.parameter_sizes)
        ):
            pointer = int(
                gdb.parse_and_eval(f"*(void**)({parameters:#x}+{8 * index})")
            )
            gdb.write(f"{self.label}_{name} address={pointer:#x}\n")
            gdb.execute(f"x/{size}gf {pointer:#x}")
        ResidualFinish(
            gdb.newest_frame(), self.label, residuals, residual_count
        )
        return False


ComponentEntry(
    0x5555557e2250,
    "ACCELERATION",
    (
        "ROTATION",
        "VELOCITY_FIRST",
        "VELOCITY_SECOND",
        "GRAVITY_DIRECTION",
        "GRAVITY_MAGNITUDE",
        "IMU_ORIENTATION",
    ),
    (4, 3, 3, 3, 1, 4),
)
ComponentEntry(
    0x5555557ddfb0,
    "DELTA_ROTATION",
    ("ROTATION_FIRST", "ROTATION_SECOND", "IMU_ORIENTATION"),
    (4, 4, 4),
)
end

continue
