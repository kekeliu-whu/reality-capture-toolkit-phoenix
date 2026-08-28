set pagination off
set confirm off
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_fast_velocity_steps.log
set logging overwrite on
set logging enabled on
starti

python
import gdb


IMAGE_BASE = 0x555555554000
HELPER = IMAGE_BASE + 0x2586B0
INTERVAL_RETURN = IMAGE_BASE + 0x237923
STEP_INPUT = IMAGE_BASE + 0x258A48
STEP_OUTPUT = IMAGE_BASE + 0x258B3C
STATE_SIGNATURE_CHECK_CALL = IMAGE_BASE + 0x14BB16
STATE_SIGNATURE_CHECK_RETURN = IMAGE_BASE + 0x14BB1B
DATA_SIGNATURE_CHECK_CALL = IMAGE_BASE + 0x14BB26
DATA_SIGNATURE_CHECK_RETURN = IMAGE_BASE + 0x14BB2B


def words(address, count):
    return [
        int(gdb.parse_and_eval(f"*(unsigned long long*)({address:#x}+{8 * index})"))
        for index in range(count)
    ]


class StepInput(gdb.Breakpoint):
    def __init__(self):
        super().__init__(
            f"*{STEP_INPUT:#x}",
            type=gdb.BP_HARDWARE_BREAKPOINT,
            internal=False,
        )
        self.index = 0

    def stop(self):
        stack = int(gdb.parse_and_eval("$rsp"))
        gdb.write(
            "STEP_INPUT " + str(self.index) + " "
            + " ".join(f"{value:016x}" for value in words(stack, 88))
            + "\n"
        )
        return False


class StepOutput(gdb.Breakpoint):
    def __init__(self, input_breakpoint):
        super().__init__(f"*{STEP_OUTPUT:#x}", internal=True)
        self.enabled = False
        self.input_breakpoint = input_breakpoint

    def stop(self):
        output = int(gdb.parse_and_eval("$r12"))
        stack = int(gdb.parse_and_eval("$rsp"))
        index = self.input_breakpoint.index
        gdb.write(
            "STEP_STACK " + str(index) + " "
            + " ".join(f"{value:016x}" for value in words(stack, 88))
            + "\n"
        )
        gdb.write(
            "STEP_OUTPUT " + str(index) + " "
            + " ".join(f"{value:016x}" for value in words(output, 10))
            + "\n"
        )
        self.input_breakpoint.index += 1
        return False


class AcceptSignedInput(gdb.Breakpoint):
    """Permit a frozen official protobuf to be replayed after it was copied."""

    def __init__(self, call_address, return_address):
        super().__init__(f"*{call_address:#x}", internal=False)
        self.return_address = return_address

    def stop(self):
        gdb.execute("set $rax = 1")
        gdb.execute(f"set $pc = {self.return_address:#x}")
        gdb.write(f"ACCEPT_SIGNED_INPUT {int(gdb.parse_and_eval('$pc')):#x}\n")
        self.enabled = False
        return False


class IntegratorEntry(gdb.Breakpoint):
    def __init__(self, input_breakpoint, output_breakpoint):
        super().__init__(f"*{HELPER:#x}", internal=False)
        self.input_breakpoint = input_breakpoint
        self.output_breakpoint = output_breakpoint
        self.captured = False

    def stop(self):
        stack = int(gdb.parse_and_eval("$rsp"))
        return_address = int(gdb.parse_and_eval(f"*(void**){stack:#x}"))
        if self.captured or return_address != INTERVAL_RETURN:
            return False
        self.captured = True
        self.enabled = False
        self.input_breakpoint.enabled = True
        self.output_breakpoint.enabled = True
        finish.enabled = True
        return False


class Finish(gdb.Breakpoint):
    def __init__(self, input_breakpoint, output_breakpoint):
        super().__init__(f"*{INTERVAL_RETURN:#x}", internal=True)
        self.enabled = False
        self.input_breakpoint = input_breakpoint
        self.output_breakpoint = output_breakpoint

    def stop(self):
        self.input_breakpoint.enabled = False
        self.output_breakpoint.enabled = False
        gdb.execute("set logging enabled off")
        gdb.execute("quit")
        return False


step_input = StepInput()
step_output = StepOutput(step_input)
finish = Finish(step_input, step_output)
signed_input_breakpoints = [
    AcceptSignedInput(STATE_SIGNATURE_CHECK_CALL, STATE_SIGNATURE_CHECK_RETURN),
    AcceptSignedInput(DATA_SIGNATURE_CHECK_CALL, DATA_SIGNATURE_CHECK_RETURN),
]
IntegratorEntry(step_input, step_output)
end

continue
