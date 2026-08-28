set pagination off
set confirm off
set breakpoint pending on
starti
python
import gdb
import json
import math
import struct

OUTPUT = "/tmp/navvis_vendor_aic_steps.jsonl"
MAX_EVENTS = 40
events = 0

open(OUTPUT, "w").close()

def f32(address):
    data = bytes(gdb.selected_inferior().read_memory(address, 4))
    return struct.unpack("<f", data)[0]

def append(record):
    with open(OUTPUT, "a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")

class ReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, stage, pointers, before, ring):
        super().__init__(gdb.newest_frame(), internal=True)
        self.stage = stage
        self.pointers = pointers
        self.before = before
        self.ring = ring

    def stop(self):
        global events
        after = [f32(pointer) for pointer in self.pointers]
        append({
            "event": events,
            "stage": self.stage,
            "ring": self.ring,
            "before": self.before,
            "after": after,
        })
        events += 1
        if events >= MAX_EVENTS:
            gdb.write("captured %d AIC stage calls\n" % events)
            gdb.execute("quit")
        return False

class UnaryBreakpoint(gdb.Breakpoint):
    def __init__(self, symbol, stage):
        super().__init__(symbol, internal=True)
        self.stage = stage

    def stop(self):
        pointer = int(gdb.parse_and_eval("$rsi"))
        ring = int(gdb.parse_and_eval("$edx"))
        ReturnBreakpoint(self.stage, [pointer], [f32(pointer)], ring)
        return False

class RangeGainBiasBreakpoint(gdb.Breakpoint):
    def __init__(self, spec="_ZNK6navvis6sensor21LaserModelMultilayerTIfE28applyRangeGainBiasCorrectionERfif"):
        super().__init__(spec, internal=True)

    def stop(self):
        pointer = int(gdb.parse_and_eval("$rsi"))
        ring = int(gdb.parse_and_eval("$edx"))
        temperature = float(gdb.parse_and_eval("$xmm0.v4_float[0]"))
        before = [f32(pointer), temperature]
        ReturnBreakpoint("range_gain_bias", [pointer], before, ring)
        return False

class RayParallaxBreakpoint(gdb.Breakpoint):
    def __init__(self, spec="_ZNK6navvis6sensor21LaserModelMultilayerTIfE26applyRayParallaxCorrectionERfS3_RKfi"):
        super().__init__(spec, internal=True)

    def stop(self):
        x = int(gdb.parse_and_eval("$rsi"))
        y = int(gdb.parse_and_eval("$rdx"))
        azimuth = int(gdb.parse_and_eval("$rcx"))
        ring = int(gdb.parse_and_eval("$r8d"))
        ReturnBreakpoint(
            "ray_parallax", [x, y], [f32(x), f32(y), f32(azimuth)], ring
        )
        return False

class TranslationBreakpoint(gdb.Breakpoint):
    def __init__(self, spec="_ZNK6navvis6sensor21LaserModelMultilayerTIfE26applyTranslationCorrectionERfS3_S3_i"):
        super().__init__(spec, internal=True)

    def stop(self):
        x = int(gdb.parse_and_eval("$rsi"))
        y = int(gdb.parse_and_eval("$rdx"))
        z = int(gdb.parse_and_eval("$rcx"))
        ring = int(gdb.parse_and_eval("$r8d"))
        ReturnBreakpoint("translation", [x, y, z], [f32(x), f32(y), f32(z)], ring)
        return False

installed_sensor = False
installed_dataset = False

def image_base(path_fragment):
    pid = gdb.selected_inferior().pid
    if not pid:
        return None
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and path_fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    return None

def install_dataset_breakpoints():
    global installed_dataset
    if installed_dataset:
        return
    base = image_base("liblibnavvis_dataset.so.4.85")
    if base is None:
        return
    installed_dataset = True
    UnaryBreakpoint("*%#x" % (base + 0x11b9a0), "excentricity")
    UnaryBreakpoint("*%#x" % (base + 0x11af40), "azimuth")
    UnaryBreakpoint("*%#x" % (base + 0x11aed0), "cone")
    RangeGainBiasBreakpoint("*%#x" % (base + 0x13b950))
    UnaryBreakpoint("*%#x" % (base + 0x11bd10), "range_spline")
    RayParallaxBreakpoint("*%#x" % (base + 0x11ba40))
    TranslationBreakpoint("*%#x" % (base + 0x11afa0))
    gdb.write("installed dataset AIC breakpoints at %#x\n" % base)

def install_breakpoints(event=None):
    global installed_sensor
    names = [objfile.filename or "" for objfile in gdb.current_progspace().objfiles()]
    if not installed_sensor and any("liblibnavvis_sensor.so" in name for name in names):
        installed_sensor = True
        UnaryBreakpoint(
            "_ZNK6navvis6sensor21LaserModelMultilayerTIfE27applyExcentricityCorrectionERfi",
            "excentricity",
        )
        UnaryBreakpoint(
            "_ZNK6navvis6sensor21LaserModelMultilayerTIfE22applyAzimuthCorrectionERfi",
            "azimuth",
        )
        UnaryBreakpoint(
            "_ZNK6navvis6sensor21LaserModelMultilayerTIfE24applyConeAngleCorrectionERfi",
            "cone",
        )
        RangeGainBiasBreakpoint()
        UnaryBreakpoint(
            "_ZNK6navvis6sensor21LaserModelMultilayerTIfE30applyRangeBiasSplineCorrectionERfi",
            "range_spline",
        )
        RayParallaxBreakpoint()
        TranslationBreakpoint()
        gdb.write("installed sensor AIC breakpoints\n")
    if any("liblibnavvis_dataset.so" in name for name in names):
        install_dataset_breakpoints()

gdb.events.new_objfile.connect(install_breakpoints)
install_breakpoints()
end
continue
