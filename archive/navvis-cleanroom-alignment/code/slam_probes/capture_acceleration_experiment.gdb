set pagination off
set confirm off
starti

python
import gdb
import os


experiment = os.environ.get("NAVVIS_ACCEL_EXPERIMENT", "baseline")


def write_double(address, value):
    gdb.execute(f"set {{double}}{address:#x} = {value:.17g}")


def write_vector(address, values):
    for index, value in enumerate(values):
        write_double(address + 8 * index, value)


class ResidualFinish(gdb.FinishBreakpoint):
    def __init__(self, frame, residual_pointer, residual_count):
        super().__init__(frame, internal=True)
        self.residual_pointer = residual_pointer
        self.residual_count = residual_count

    def stop(self):
        gdb.write(f"ACCELERATION_EXPERIMENT={experiment}\n")
        gdb.write("ACCELERATION_RESIDUALS\n")
        gdb.execute(f"x/{self.residual_count}gf {self.residual_pointer:#x}")
        gdb.execute("quit")
        return False


class AccelerationEntry(gdb.Breakpoint):
    def stop(self):
        self.enabled = False
        this_pointer = int(gdb.parse_and_eval("$rdi"))
        parameters = int(gdb.parse_and_eval("$rsi"))
        residuals = int(gdb.parse_and_eval("$rdx"))
        residual_count = int(gdb.parse_and_eval(f"*(int*)({this_pointer:#x}+32)"))
        functor = int(gdb.parse_and_eval(f"*(void**)({this_pointer:#x}+40)"))
        pointers = [
            int(gdb.parse_and_eval(f"*(void**)({parameters:#x}+{8 * index})"))
            for index in range(6)
        ]

        if experiment == "zero_measurement":
            write_vector(functor, (0.0, 0.0, 0.0))
        elif experiment == "identity_imu":
            write_vector(pointers[5], (0.0, 0.0, 0.0, 1.0))
        elif experiment == "identity_tracking":
            write_vector(pointers[0], (0.0, 0.0, 0.0, 1.0))
        elif experiment == "zero_gravity":
            write_double(pointers[4], 0.0)
        elif experiment == "zero_positions":
            for pointer in pointers[1:4]:
                write_vector(pointer, (0.0, 0.0, 0.0))
        elif experiment != "baseline":
            raise gdb.GdbError(f"unknown acceleration experiment: {experiment}")

        gdb.write("ACCELERATION_INPUTS_AFTER_MUTATION\n")
        gdb.execute(f"x/5gf {functor:#x}")
        for pointer, size in zip(pointers, (4, 3, 3, 3, 1, 4)):
            gdb.execute(f"x/{size}gf {pointer:#x}")
        ResidualFinish(gdb.newest_frame(), residuals, residual_count)
        return False


AccelerationEntry("*0x5555557e2250", internal=False)
end

continue
