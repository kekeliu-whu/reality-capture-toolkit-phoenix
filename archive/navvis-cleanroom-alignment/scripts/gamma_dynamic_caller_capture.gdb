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


out_dir = pathlib.Path(os.environ["GAMMA_DYNAMIC_CALLER_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
fixed_ovs = pathlib.Path(os.environ["GAMMA_DYNAMIC_CALLER_OVS"])


def read(address, size):
    return gdb.selected_inferior().read_memory(address, size).tobytes()


def u64(address):
    return struct.unpack("<Q", read(address, 8))[0]


class ExposureOvsInjection(gdb.Breakpoint):
    def stop(self):
        vector = int(gdb.parse_and_eval("$rsi"))
        begin = u64(vector)
        end = u64(vector + 8)
        payload = fixed_ovs.read_bytes()
        if len(payload) != end - begin:
            raise gdb.GdbError(
                "fixed OVS size mismatch: %d != %d" % (len(payload), end - begin)
            )
        gdb.selected_inferior().write_memory(begin, payload)
        (out_dir / "effective_exposure_ovs.bin").write_bytes(payload)
        self.enabled = False
        return False


class DynamicCallerState(gdb.Breakpoint):
    def stop(self):
        stack = int(gdb.parse_and_eval("$rsp"))
        normalized_begin = int(gdb.parse_and_eval("$r14"))
        records_begin = int(gdb.parse_and_eval("$rbp"))
        count = u64(stack + 0x28)
        count_as_double = struct.unpack("<d", read(stack, 8))[0]
        total_scale = struct.unpack("<d", read(stack + 0x38, 8))[0]
        denominator = float(gdb.parse_and_eval("$xmm14.v2_double[0]"))
        normalized = read(normalized_begin, count * 8)
        records = read(records_begin, count * 32)
        (out_dir / "vendor_normalized_weights.bin").write_bytes(normalized)
        (out_dir / "vendor_dynamic_records.bin").write_bytes(records)
        values = struct.unpack("<%dd" % count, normalized)
        record_rows = []
        for index in range(count):
            base = index * 32
            raw_view = struct.unpack_from("<I", records, base + 16)[0]
            low = records[base + 24]
            high = records[base + 25]
            record_rows.append({
                "index": index,
                "raw_view": raw_view,
                "low": low,
                "high": high,
                "normalized_weight_hex": values[index].hex(),
            })
        metadata = {
            "breakpoint_static": "0x1e66db",
            "count": count,
            "count_as_double": count_as_double,
            "count_as_double_hex": count_as_double.hex(),
            "l1_denominator": denominator,
            "l1_denominator_hex": denominator.hex(),
            "total_scale": total_scale,
            "total_scale_hex": total_scale.hex(),
            "average_scale": total_scale / count_as_double,
            "average_scale_hex": (total_scale / count_as_double).hex(),
            "normalized_sum": sum(values),
            "normalized_sum_hex": sum(values).hex(),
            "sha256": {
                "normalized_weights": hashlib.sha256(normalized).hexdigest(),
                "dynamic_records": hashlib.sha256(records).hexdigest(),
            },
            "records": record_rows,
        }
        (out_dir / "vendor_dynamic_caller.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.enabled = False
        return False


# PIE load bias is 0x555555554000 with ASLR disabled for this executable.
ExposureOvsInjection("*0x55555573ea70", internal=True)
DynamicCallerState("*0x55555573a6db", internal=True)
end

run
quit
