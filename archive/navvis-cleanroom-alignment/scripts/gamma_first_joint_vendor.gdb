set pagination off
set confirm off
set breakpoint pending on
set disable-randomization on
set print thread-events off

python
import hashlib
import json
import os
import pathlib
import struct

import gdb


out_dir = pathlib.Path(os.environ["GAMMA_FIRST_JOINT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
ovs_path = pathlib.Path(os.environ["GAMMA_FIRST_JOINT_OVS"])


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def write(address, payload):
    gdb.selected_inferior().write_memory(address, payload)


class ExposureBreakpoint(gdb.Breakpoint):
    def stop(self):
        vector = int(gdb.parse_and_eval("$rsi"))
        begin, end = struct.unpack("<QQ", read(vector, 16))
        original = read(begin, end - begin)
        effective = ovs_path.read_bytes()
        if len(original) != len(effective):
            raise gdb.GdbError("OVS size mismatch")
        write(begin, effective)
        (out_dir / "effective_exposure_ovs.bin").write_bytes(effective)
        (out_dir / "ovs_patch.json").write_text(
            json.dumps({
                "records": len(effective) // 40,
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "effective_sha256": hashlib.sha256(effective).hexdigest(),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.enabled = False
        return False


class SolveBreakpoint(gdb.Breakpoint):
    def stop(self):
        options = int(gdb.parse_and_eval("$rdi"))
        write(options + 120, struct.pack("<i", 1))
        self.enabled = False
        return False


class JointReturn(gdb.FinishBreakpoint):
    def __init__(self, frame, cost, parameters, residuals, jacobians, block_count,
                 residual_count):
        super().__init__(frame, internal=True)
        self.cost = cost
        self.parameters = parameters
        self.residuals = residuals
        self.jacobians = jacobians
        self.block_count = block_count
        self.residual_count = residual_count

    def stop(self):
        parameter_payload = bytearray()
        jacobian_payload = bytearray()
        parameter_addresses = []
        jacobian_addresses = []
        for block in range(self.block_count):
            parameter_address = struct.unpack(
                "<Q", read(self.parameters + block * 8, 8)
            )[0]
            jacobian_address = struct.unpack(
                "<Q", read(self.jacobians + block * 8, 8)
            )[0]
            parameter_addresses.append(hex(parameter_address))
            jacobian_addresses.append(hex(jacobian_address))
            parameter_payload.extend(read(parameter_address, 16))
            jacobian_payload.extend(read(jacobian_address, self.residual_count * 16))
        residual_payload = read(self.residuals, self.residual_count * 8)
        (out_dir / "raw_parameters.bin").write_bytes(parameter_payload)
        (out_dir / "raw_residuals.bin").write_bytes(residual_payload)
        (out_dir / "raw_jacobians_block_major.bin").write_bytes(jacobian_payload)
        metadata = {
            "cost_function_address": hex(self.cost),
            "block_count": self.block_count,
            "residual_count": self.residual_count,
            "parameter_addresses": parameter_addresses,
            "jacobian_addresses": jacobian_addresses,
            "parameters_hex": [
                value.hex()
                for value in struct.unpack("<%dd" % (2 * self.block_count), parameter_payload)
            ],
            "residuals_hex": [
                value.hex()
                for value in struct.unpack("<%dd" % self.residual_count, residual_payload)
            ],
            "sha256": {
                "parameters": hashlib.sha256(parameter_payload).hexdigest(),
                "residuals": hashlib.sha256(residual_payload).hexdigest(),
                "jacobians_block_major": hashlib.sha256(jacobian_payload).hexdigest(),
            },
        }
        (out_dir / "first_joint_raw.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return True


class FirstJoint(gdb.Breakpoint):
    def stop(self):
        cost = int(gdb.parse_and_eval("$rdi"))
        parameters = int(gdb.parse_and_eval("$rsi"))
        residuals = int(gdb.parse_and_eval("$rdx"))
        jacobians = int(gdb.parse_and_eval("$rcx"))
        if jacobians == 0:
            return False
        begin, end = struct.unpack("<QQ", read(cost + 8, 16))
        block_count = (end - begin) // 4
        residual_count = struct.unpack("<i", read(cost + 0x20, 4))[0]
        if block_count < 2 or block_count > 5 or residual_count != block_count:
            raise gdb.GdbError(
                "unexpected first Joint dimensions: blocks=%d residuals=%d"
                % (block_count, residual_count)
            )
        JointReturn(
            gdb.newest_frame(), cost, parameters, residuals, jacobians,
            block_count, residual_count
        )
        self.enabled = False
        return False


# Hash-frozen nv_colorcloud Build ID a7586f518009434f5e97891f897aea42675f26a0.
ExposureBreakpoint("*0x55555573ea70", internal=True)
SolveBreakpoint("*0x555555a7e890", internal=True)
FirstJoint("*0x555555742070", internal=True)
end

run
quit
