set pagination off
set confirm off
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_helper_probe/integrator.log
set logging overwrite on
set logging enabled on
starti

python
import gdb


IMAGE_BASE = 0x555555554000
HELPER = IMAGE_BASE + 0x2586B0
TARGET_RETURNS = {
    IMAGE_BASE + 0x237923: "INTERVAL",
    IMAGE_BASE + 0x237BFA: "FIRST_TO_MIDPOINT",
    IMAGE_BASE + 0x237CAB: "MIDPOINT_TO_MIDPOINT",
}


def dump_words(label, address, count=24):
    gdb.write(f"{label} address={address:#x}\n")
    if address < 0x10000:
        return
    try:
        gdb.execute(f"x/{count}gx {address:#x}")
    except gdb.error as error:
        gdb.write(f"{label}_ERROR {error}\n")


class IntegratorEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{HELPER:#x}", internal=False)
        self.captured = set()
        self.outputs = {}

    def stop(self):
        stack = int(gdb.parse_and_eval("$rsp"))
        return_address = int(gdb.parse_and_eval(f"*(void**){stack:#x}"))
        label = TARGET_RETURNS.get(return_address)
        if label is None or label in self.captured:
            return False
        self.captured.add(label)
        registers = {
            name: int(gdb.parse_and_eval(f"${name}"))
            for name in ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
        }
        stack_arguments = [
            int(gdb.parse_and_eval(f"*(void**)({stack:#x}+{8 * index})"))
            for index in range(1, 7)
        ]
        gdb.write(
            f"{label}_ENTRY return={return_address:#x} "
            + " ".join(f"{name}={value:#x}" for name, value in registers.items())
            + " "
            + " ".join(
                f"stack{index}={value:#x}"
                for index, value in enumerate(stack_arguments)
            )
            + "\n"
        )
        self.outputs[label] = registers["rdi"]
        for name in ("r8", "r9"):
            dump_words(f"{label}_{name.upper()}", registers[name], 12)
        for index, value in enumerate(stack_arguments):
            dump_words(f"{label}_STACK{index}", value, 12)
        return False


class IntegratorReturn(gdb.Breakpoint):
    def __init__(self, address, label, owner):
        super().__init__(f"*{address:#x}", internal=False)
        self.label = label
        self.owner = owner

    def stop(self):
        output = self.owner.outputs.get(self.label)
        if output is None:
            return False
        self.enabled = False
        gdb.write(f"{self.label}_RETURN\n")
        dump_words(f"{self.label}_OUTPUT", output, 40)
        if all(not breakpoint.enabled for breakpoint in return_breakpoints):
            gdb.execute("set logging enabled off")
            gdb.execute("quit")
        return False


entry = IntegratorEntry()
return_breakpoints = [
    IntegratorReturn(address, label, entry)
    for address, label in TARGET_RETURNS.items()
]
end

continue
