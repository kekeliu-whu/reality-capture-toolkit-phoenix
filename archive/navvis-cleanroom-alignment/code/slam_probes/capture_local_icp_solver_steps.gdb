set pagination off
set confirm off
set breakpoint pending on
set disable-randomization on
starti
python
import gdb
import json
import os
import struct
from pathlib import Path


OUTPUT = Path(os.environ.get(
    "NAVVIS_LOCAL_ICP_SOLVER_DIR", "/tmp/navvis-local-icp-solver"
))
TARGET_MATCH_INDEX = int(os.environ.get("NAVVIS_LOCAL_ICP_MATCH_INDEX", "0"))
TARGET_PREDICATE_TARGET = int(os.environ.get(
    "NAVVIS_LOCAL_ICP_PREDICATE_TARGET", "1624"
))
TARGET_PREDICATE_SOURCE = int(os.environ.get(
    "NAVVIS_LOCAL_ICP_PREDICATE_SOURCE", "8225"
))
TARGET_PREDICATE_ITERATION = int(os.environ.get(
    "NAVVIS_LOCAL_ICP_PREDICATE_ITERATION", "7"
))
EXECUTABLE_FRAGMENT = os.environ.get(
    "NAVVIS_LOCAL_ICP_EXECUTABLE_FRAGMENT", "surveyorslam_processing_node"
)


def configured_offset(name, default):
    return int(os.environ.get(name, hex(default)), 0)


MATCH_OFFSET = configured_offset("NAVVIS_LOCAL_ICP_MATCH_OFFSET", 0x6C4660)
SOLVER_RETURN_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_SOLVER_RETURN_OFFSET", 0x6CEB90
)
COMPOSE_RETURN_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_COMPOSE_RETURN_OFFSET", 0x6CEBA8
)
HESSIAN_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_HESSIAN_READY_OFFSET", 0x6DE490
)
RHS_READY_OFFSET = configured_offset("NAVVIS_LOCAL_ICP_RHS_READY_OFFSET", 0x6DE4A3)
DELTA_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_DELTA_READY_OFFSET", 0x6DE4A8
)
KERNEL_INVERSE_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_KERNEL_INVERSE_READY_OFFSET", 0x6DE6DF
)
KERNEL_FIRST_COMPOSE_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_KERNEL_FIRST_COMPOSE_READY_OFFSET", 0x6DE6ED
)
KERNEL_SECOND_COMPOSE_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_KERNEL_SECOND_COMPOSE_READY_OFFSET", 0x6DE703
)
INCREMENT_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_INCREMENT_READY_OFFSET", 0x6DF5B5
)
NORMALIZED_POSE_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_NORMALIZED_POSE_READY_OFFSET", 0x6DF672
)
GN_ENTRY_OFFSET = configured_offset("NAVVIS_LOCAL_ICP_GN_ENTRY_OFFSET", 0x6DD6B0)
OUTER_TRANSFORM_CALL_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_OUTER_TRANSFORM_CALL_OFFSET", 0x6CEB33
)
NORMALIZATION_ENTRY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_NORMALIZATION_ENTRY_OFFSET", 0x6CF030
)
NORMALIZATION_ALIGNMENT_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_NORMALIZATION_ALIGNMENT_READY_OFFSET", 0x6CFE7F
)
NORMALIZATION_PRODUCT_INPUT_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_NORMALIZATION_PRODUCT_INPUT_READY_OFFSET", 0x6CFEDC
)
NORMALIZATION_PRODUCT_READY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_NORMALIZATION_PRODUCT_READY_OFFSET", 0x6CFF69
)
FILTER_PREDICATE_DECISION_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_FILTER_PREDICATE_DECISION_OFFSET", 0x6C962D
)
FILTER_PREDICATE_GEOMETRY_OFFSET = configured_offset(
    "NAVVIS_LOCAL_ICP_FILTER_PREDICATE_GEOMETRY_OFFSET", 0x6C9601
)
DEFAULT_FILTER_PREDICATE_CALL_OFFSETS = (
    0x6CE377,
    0x6CE392,
    0x6CE3AD,
    0x6CE3D2,
    0x6CE438,
    0x6CE45B,
    0x6CE492,
    0x6CE4D2,
    0x6CE50D,
    0x6CE54B,
    0x6CE58A,
)
FILTER_PREDICATE_CALL_OFFSETS = tuple(
    int(value, 0)
    for value in os.environ.get(
        "NAVVIS_LOCAL_ICP_FILTER_PREDICATE_CALL_OFFSETS",
        ",".join(hex(value) for value in DEFAULT_FILTER_PREDICATE_CALL_OFFSETS),
    ).split(",")
)
FILTER_PREDICATE_RETURN_OFFSETS = tuple(
    offset + 3 for offset in FILTER_PREDICATE_CALL_OFFSETS
)


def image_base(fragment):
    pid = gdb.selected_inferior().pid
    with open("/proc/%d/maps" % pid) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) >= 6 and fields[2] == "00000000" and fragment in fields[-1]:
                return int(fields[0].split("-")[0], 16)
    raise RuntimeError("executable mapping was not found")


def memory(address, size):
    return bytes(gdb.selected_inferior().read_memory(address, size))


def decode_pose(address):
    payload = memory(address, 64)
    values = struct.unpack("<8d", payload)
    return {
        "address": address,
        "translation": list(values[:3]),
        "padding": values[3],
        "quaternion_xyzw": list(values[4:]),
        "hex": payload.hex(),
    }


def decode_doubles(address, count):
    return list(struct.unpack("<%dd" % count, memory(address, 8 * count)))


BASE = image_base(EXECUTABLE_FRAGMENT)
OUTPUT.mkdir(parents=True, exist_ok=True)
state = {
    "in_match": False,
    "match_thread": None,
    "match_seen": 0,
    "steps": [],
    "kernels": [],
    "gn_entries": [],
    "normalizations": [],
    "iteration_transforms": [],
    "predicate_hits": [],
}
pending_by_thread = {}
pending_kernel_by_thread = {}
pending_predicate_by_thread = {}
predicate_breakpoints = []
predicate_decision_breakpoint = None
predicate_geometry_breakpoint = None


def write_trace():
    payload = {
        "executable_base": BASE,
        "target_match_index": TARGET_MATCH_INDEX,
        "match_offset": MATCH_OFFSET,
        "solver_return_offset": SOLVER_RETURN_OFFSET,
        "compose_return_offset": COMPOSE_RETURN_OFFSET,
        "steps": state["steps"],
        "kernels": state["kernels"],
        "gn_entries": state["gn_entries"],
        "normalizations": state["normalizations"],
        "iteration_transforms": state["iteration_transforms"],
        "predicate_hits": state["predicate_hits"],
    }
    (OUTPUT / "trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


class MatchReturn(gdb.FinishBreakpoint):
    def stop(self):
        write_trace()
        gdb.write("captured %d local ICP solver steps\n" % len(state["steps"]))
        gdb.execute("quit")
        return False


class MatchBreakpoint(gdb.Breakpoint):
    def stop(self):
        if state["in_match"]:
            return False
        match_index = state["match_seen"]
        state["match_seen"] += 1
        if match_index != TARGET_MATCH_INDEX:
            return False
        state["in_match"] = True
        state["match_thread"] = gdb.selected_thread().num
        if TARGET_PREDICATE_ITERATION < 0:
            for breakpoint in predicate_breakpoints:
                breakpoint.enabled = True
        MatchReturn(gdb.newest_frame(), internal=True)
        return False


class SolverReturn(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if not state["in_match"] or thread != state["match_thread"]:
            return False
        stack = int(gdb.parse_and_eval("$rsp"))
        pose_before_address = struct.unpack("<Q", memory(stack + 8, 8))[0]
        step = {
            "index": len(state["steps"]),
            "binary_iteration": int(gdb.parse_and_eval("$rbx")),
            "increment": decode_pose(stack + 0x120),
            "pose_before": decode_pose(pose_before_address),
            "pose_after": None,
        }
        state["steps"].append(step)
        pending_by_thread[thread] = step
        write_trace()
        return False


class ComposeReturn(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        step = pending_by_thread.pop(thread, None)
        if step is None:
            return False
        stack = int(gdb.parse_and_eval("$rsp"))
        step["pose_after"] = decode_pose(stack + 0x160)
        write_trace()
        return False


class HessianReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if not state["in_match"] or thread != state["match_thread"]:
            return False
        address = int(gdb.parse_and_eval("$rsi"))
        kernel = {
            "index": len(state["kernels"]),
            "hessian_address": address,
            "hessian": decode_doubles(address, 36),
            "rhs": None,
            "delta": None,
            "inverse_normalization": None,
            "normalized_delta_pose": None,
            "first_compose": None,
            "second_compose_before_normalization": None,
            "increment": None,
            "normalized_pose": None,
        }
        state["kernels"].append(kernel)
        pending_kernel_by_thread[thread] = kernel
        write_trace()
        return False


class GnEntry(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if not state["in_match"] or thread != state["match_thread"]:
            return False
        solver = int(gdb.parse_and_eval("$rsi"))
        state["gn_entries"].append({
            "index": len(state["gn_entries"]),
            "solver_address": solver,
            "normalization": decode_pose(solver + 0x20),
        })
        write_trace()
        return False


def decode_xmm(name):
    return [
        float(gdb.parse_and_eval("$%s.v2_double[%d]" % (name, lane)))
        for lane in range(2)
    ]


class NormalizationEntry(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if not state["in_match"] or thread != state["match_thread"]:
            return False
        input_address = int(gdb.parse_and_eval("$rsi"))
        state["normalizations"].append({
            "index": len(state["normalizations"]),
            "input_address": input_address,
            "input_quaternion_xyzw": decode_doubles(input_address, 4),
            "raw_gravity": None,
            "normalized_gravity": None,
            "gravity_alignment_xyzw": None,
            "product_rhs_xyzw": None,
            "product_xyzw": None,
        })
        write_trace()
        return False


class NormalizationAlignmentReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if (not state["normalizations"] or not state["in_match"] or
                thread != state["match_thread"]):
            return False
        stack = int(gdb.parse_and_eval("$rsp"))
        record = state["normalizations"][-1]
        record["raw_gravity"] = (
            decode_doubles(stack + 0x120, 2) +
            decode_doubles(stack + 0x130, 1)
        )
        record["normalized_gravity"] = (
            decode_doubles(stack + 0x140, 2) +
            decode_doubles(stack + 0x150, 1)
        )
        record["gravity_alignment_xyzw"] = (
            decode_xmm("xmm1") + decode_doubles(stack + 0x170, 2)
        )
        write_trace()
        return False


class NormalizationProductInputReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if (not state["normalizations"] or not state["in_match"] or
                thread != state["match_thread"]):
            return False
        state["normalizations"][-1]["product_rhs_xyzw"] = (
            decode_xmm("xmm1") + decode_xmm("xmm9")
        )
        write_trace()
        return False


class NormalizationProductReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if (not state["normalizations"] or not state["in_match"] or
                thread != state["match_thread"]):
            return False
        state["normalizations"][-1]["product_xyzw"] = (
            decode_xmm("xmm15") + decode_xmm("xmm4")
        )
        write_trace()
        return False


class OuterTransformCall(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if not state["in_match"] or thread != state["match_thread"]:
            return False
        index = len(state["iteration_transforms"])
        state["iteration_transforms"].append({
            "index": index,
            "pose": decode_pose(int(gdb.parse_and_eval("$rdx"))),
        })
        if TARGET_PREDICATE_ITERATION < 0:
            for breakpoint in predicate_breakpoints:
                breakpoint.enabled = True
        elif index == TARGET_PREDICATE_ITERATION:
            for breakpoint in predicate_breakpoints:
                breakpoint.enabled = True
        elif index == TARGET_PREDICATE_ITERATION + 1:
            for breakpoint in predicate_breakpoints:
                breakpoint.enabled = False
        write_trace()
        return False


class FilterPredicateCall(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        if not state["in_match"] or thread != state["match_thread"]:
            return False
        record_address = int(gdb.parse_and_eval("$rsi"))
        target, source, distance = struct.unpack(
            "<QQf", memory(record_address, 20)
        )
        if ((TARGET_PREDICATE_TARGET >= 0 and
             target != TARGET_PREDICATE_TARGET) or
                (TARGET_PREDICATE_SOURCE >= 0 and
                 source != TARGET_PREDICATE_SOURCE)):
            return False
        filter_address = int(gdb.parse_and_eval("$rdi"))
        hit = {
            "index": len(state["predicate_hits"]),
            "binary_iteration": int(gdb.parse_and_eval("$ecx")),
            "target": target,
            "source": source,
            "distance_squared": distance,
            "filter_address": filter_address,
            "filter_hex": memory(filter_address, 128).hex(),
            "call_address": int(gdb.parse_and_eval("$pc")),
            "function_address": struct.unpack(
                "<Q", memory(int(gdb.parse_and_eval("$rax")) + 0x10, 8)
            )[0],
            "predicate_pose": decode_pose(
                int(gdb.parse_and_eval("$rdx"))
            ),
            "plane_residual": None,
            "plane_threshold": None,
            "geometry_registers": None,
            "result": None,
        }
        state["predicate_hits"].append(hit)
        pending_predicate_by_thread[thread] = hit
        if TARGET_PREDICATE_ITERATION < 0:
            # Schedule probes need only one representative correspondence per
            # outer iteration.  Re-enable at the next outer transform instead
            # of trapping every remaining source point in this iteration.
            for breakpoint in predicate_breakpoints:
                breakpoint.enabled = False
        predicate_decision_breakpoint.enabled = True
        predicate_geometry_breakpoint.enabled = True
        write_trace()
        return False


class FilterPredicateReturn(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        hit = pending_predicate_by_thread.pop(thread, None)
        if hit is None:
            return False
        hit["result"] = int(gdb.parse_and_eval("$al"))
        write_trace()
        return False


class FilterPredicateDecision(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        hit = pending_predicate_by_thread.get(thread)
        if hit is None:
            return False
        hit["plane_residual"] = float(
            gdb.parse_and_eval("$xmm3.v4_float[0]")
        )
        hit["plane_threshold"] = float(
            gdb.parse_and_eval("$xmm2.v4_float[0]")
        )
        self.enabled = False
        write_trace()
        return False


class FilterPredicateGeometry(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        hit = pending_predicate_by_thread.get(thread)
        if hit is None:
            return False
        hit["geometry_registers"] = {
            name: float(gdb.parse_and_eval("$%s.v4_float[0]" % name))
            for name in (
                "xmm0", "xmm1", "xmm2", "xmm3", "xmm6", "xmm9",
                "xmm11", "xmm12", "xmm15"
            )
        }
        self.enabled = False
        write_trace()
        return False


class RhsReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        kernel = pending_kernel_by_thread.get(thread)
        if kernel is None:
            return False
        address = int(gdb.parse_and_eval("$rsi"))
        kernel["rhs_address"] = address
        kernel["rhs"] = decode_doubles(address, 6)
        write_trace()
        return False


class DeltaReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        kernel = pending_kernel_by_thread.get(thread)
        if kernel is None:
            return False
        address = int(gdb.parse_and_eval("$r15"))
        kernel["delta_address"] = address
        kernel["delta"] = decode_doubles(address, 6)
        write_trace()
        return False


class KernelInverseReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        kernel = pending_kernel_by_thread.get(thread)
        if kernel is None:
            return False
        kernel["inverse_normalization"] = decode_pose(
            int(gdb.parse_and_eval("$r13"))
        )
        kernel["normalized_delta_pose"] = decode_pose(
            int(gdb.parse_and_eval("$r14"))
        )
        write_trace()
        return False


class KernelFirstComposeReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        kernel = pending_kernel_by_thread.get(thread)
        if kernel is None:
            return False
        kernel["first_compose"] = decode_pose(
            int(gdb.parse_and_eval("$rbp"))
        )
        write_trace()
        return False


class KernelSecondComposeReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        kernel = pending_kernel_by_thread.get(thread)
        if kernel is None:
            return False
        kernel["second_compose_before_normalization"] = decode_pose(
            int(gdb.parse_and_eval("$rbp"))
        )
        write_trace()
        return False


class IncrementReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        kernel = pending_kernel_by_thread.get(thread)
        if kernel is None:
            return False
        address = int(gdb.parse_and_eval("$rbp")) - 0xC0
        kernel["increment"] = decode_pose(address)
        write_trace()
        return False


class NormalizedPoseReady(gdb.Breakpoint):
    def stop(self):
        thread = gdb.selected_thread().num
        kernel = pending_kernel_by_thread.pop(thread, None)
        if kernel is None:
            return False
        address = int(gdb.parse_and_eval("$r12"))
        kernel["normalized_pose"] = decode_pose(address)
        write_trace()
        return False


MatchBreakpoint("*%#x" % (BASE + MATCH_OFFSET), internal=True)
SolverReturn("*%#x" % (BASE + SOLVER_RETURN_OFFSET), internal=True)
ComposeReturn("*%#x" % (BASE + COMPOSE_RETURN_OFFSET), internal=True)
HessianReady("*%#x" % (BASE + HESSIAN_READY_OFFSET), internal=True)
RhsReady("*%#x" % (BASE + RHS_READY_OFFSET), internal=True)
DeltaReady("*%#x" % (BASE + DELTA_READY_OFFSET), internal=True)
KernelInverseReady(
    "*%#x" % (BASE + KERNEL_INVERSE_READY_OFFSET), internal=True
)
KernelFirstComposeReady(
    "*%#x" % (BASE + KERNEL_FIRST_COMPOSE_READY_OFFSET), internal=True
)
KernelSecondComposeReady(
    "*%#x" % (BASE + KERNEL_SECOND_COMPOSE_READY_OFFSET), internal=True
)
IncrementReady("*%#x" % (BASE + INCREMENT_READY_OFFSET), internal=True)
NormalizedPoseReady(
    "*%#x" % (BASE + NORMALIZED_POSE_READY_OFFSET), internal=True
)
GnEntry("*%#x" % (BASE + GN_ENTRY_OFFSET), internal=True)
NormalizationEntry(
    "*%#x" % (BASE + NORMALIZATION_ENTRY_OFFSET), internal=True
)
NormalizationAlignmentReady(
    "*%#x" % (BASE + NORMALIZATION_ALIGNMENT_READY_OFFSET), internal=True
)
NormalizationProductInputReady(
    "*%#x" % (BASE + NORMALIZATION_PRODUCT_INPUT_READY_OFFSET), internal=True
)
NormalizationProductReady(
    "*%#x" % (BASE + NORMALIZATION_PRODUCT_READY_OFFSET), internal=True
)
OuterTransformCall(
    "*%#x" % (BASE + OUTER_TRANSFORM_CALL_OFFSET), internal=True
)
for offset in FILTER_PREDICATE_CALL_OFFSETS:
    breakpoint = FilterPredicateCall(
        "*%#x" % (BASE + offset), internal=True
    )
    breakpoint.enabled = False
    predicate_breakpoints.append(breakpoint)
for offset in FILTER_PREDICATE_RETURN_OFFSETS:
    breakpoint = FilterPredicateReturn(
        "*%#x" % (BASE + offset), internal=True
    )
    breakpoint.enabled = False
    predicate_breakpoints.append(breakpoint)
predicate_decision_breakpoint = FilterPredicateDecision(
    "*%#x" % (BASE + FILTER_PREDICATE_DECISION_OFFSET), internal=True
)
predicate_decision_breakpoint.enabled = False
predicate_geometry_breakpoint = FilterPredicateGeometry(
    "*%#x" % (BASE + FILTER_PREDICATE_GEOMETRY_OFFSET), internal=True
)
predicate_geometry_breakpoint.enabled = False
gdb.write("installed local ICP solver capture at executable base %#x\n" % BASE)
end
continue
