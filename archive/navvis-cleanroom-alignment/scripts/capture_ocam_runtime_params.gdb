set pagination off
set confirm off
set print thread-events off
starti
catch load liblibnavvis_sensor.so.4.85
continue
sharedlibrary liblibnavvis_sensor

python
import gdb
import struct


symbol = "_ZNK6navvis6sensor19OCamProjectionModel18projectCCS2ICSImplERKN5Eigen6MatrixIdLi3ELi1ELi0ELi3ELi1EEERNS3_IdLi2ELi1ELi0ELi2ELi1EEERKSt8optionalIdE"


class ReturnBreakpoint(gdb.FinishBreakpoint):
    def __init__(self, frame, output_address):
        super().__init__(frame, internal=True)
        self.output_address = output_address

    def stop(self):
        inferior = gdb.selected_inferior()
        output = struct.unpack("<2d", inferior.read_memory(self.output_address, 16))
        gdb.write("runtime_ics %.17g %.17g\n" % output)
        gdb.execute("quit")
        return True


class EntryBreakpoint(gdb.Breakpoint):
    def stop(self):
        self.enabled = False
        inferior = gdb.selected_inferior()
        this_address = int(gdb.parse_and_eval("$rdi"))
        input_address = int(gdb.parse_and_eval("$rsi"))
        output_address = int(gdb.parse_and_eval("$rdx"))
        point = struct.unpack("<3d", inferior.read_memory(input_address, 24))
        coefficient_count = struct.unpack(
            "<i", inferior.read_memory(this_address + 0x4A8, 4))[0]
        principal = struct.unpack(
            "<2d", inferior.read_memory(this_address + 0x4B0, 16))
        affine = struct.unpack(
            "<3d", inferior.read_memory(this_address + 0x4C0, 24))
        minimum_normalized_z = struct.unpack(
            "<d", inferior.read_memory(this_address + 0x98, 8))[0]
        coefficients = struct.unpack(
            "<%dd" % coefficient_count,
            inferior.read_memory(this_address + 0x2A8, 8 * coefficient_count))
        gdb.write("runtime_this 0x%x\n" % this_address)
        gdb.write("runtime_ccs %.17g %.17g %.17g\n" % point)
        gdb.write("runtime_coefficient_count %d\n" % coefficient_count)
        gdb.write("runtime_coefficients %s\n" % " ".join("%.17g" % x for x in coefficients))
        gdb.write("runtime_principal %.17g %.17g\n" % principal)
        gdb.write("runtime_affine %.17g %.17g %.17g\n" % affine)
        gdb.write("runtime_minimum_normalized_z %.17g\n" % minimum_normalized_z)
        ReturnBreakpoint(gdb.newest_frame(), output_address)
        return False


EntryBreakpoint(symbol)
end

continue
