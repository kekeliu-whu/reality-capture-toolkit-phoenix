set pagination off
set confirm off
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/slam_alignment_20260827/vendor_online_stage1_layout.log
set logging overwrite on
set logging enabled on
starti

python
import gdb
import os
import struct


IMAGE_BASE = 0x555555554000
SOLVE = IMAGE_BASE + 0xEDD070
SUBMAP_TRANSLATIONS = (
    (-0.29499737350803645, -0.19634248330922155, 0.9978260344042811),
    (2.7923507044219416, -4.073808581008796, 0.30841690438902525),
)
TARGET_SOLVE = int(os.environ.get("NAVVIS_PROBE_STAGE1_SOLVE", "8"))


def read_u64(address):
    return int(gdb.parse_and_eval(f"*(unsigned long long*){address:#x}"))


def writable_mappings():
    pid = gdb.selected_inferior().pid
    mappings = []
    with open(f"/proc/{pid}/maps", "r", encoding="ascii") as stream:
        for line in stream:
            fields = line.split(maxsplit=5)
            if "w" not in fields[1]:
                continue
            first, last = (int(value, 16) for value in fields[0].split("-"))
            name = fields[5].strip() if len(fields) == 6 else ""
            mappings.append((first, last, name))
    return mappings


def containing_mapping(address, mappings):
    for first, last, name in mappings:
        if first <= address < last:
            return first, last, name
    return None


def dump_reachable_pose_storage(owner):
    inferior = gdb.selected_inferior()
    raw = inferior.read_memory(owner, 0x40).tobytes()
    gdb.write(f"CERES_PROBLEM address={owner:#x} bytes=0x40\n")
    patterns = [struct.pack("<3d", *translation) for translation in SUBMAP_TRANSLATIONS]
    mappings = writable_mappings()
    selected = set()
    for offset in range(0, len(raw), 8):
        value = struct.unpack_from("<Q", raw, offset)[0]
        mapping = containing_mapping(value, mappings)
        if mapping is not None:
            selected.add(mapping)

    # Ceres allocates its implementation and parameter blocks from the worker
    # thread's anonymous glibc arena.  Scan that mapping once; this is both much
    # faster and more reliable than following arbitrary words from the caller's
    # stack frame.
    for first, last, name in sorted(selected):
        size = last - first
        if size > 512 * 1024 * 1024:
            gdb.write(
                f"SKIP_MAPPING start={first:#x} end={last:#x} size={size} name={name!r}\n"
            )
            continue
        gdb.write(
            f"SCAN_MAPPING start={first:#x} end={last:#x} size={size} name={name!r}\n"
        )
        try:
            child = inferior.read_memory(first, size).tobytes()
        except gdb.MemoryError:
            continue
        for submap_index, pattern in enumerate(patterns):
            position = 0
            while True:
                position = child.find(pattern, position)
                if position < 0:
                    break
                address = first + position
                pose_addresses[submap_index].append(address)
                gdb.write(
                    f"POSE_STORAGE submap={submap_index} address={address:#x}\n"
                )
                start = max(first, address - 96)
                count = min(32, (last - start) // 8)
                surrounding = inferior.read_memory(start, count * 8).tobytes()
                values = struct.unpack(f"<{count}d", surrounding)
                gdb.write(
                    f"POSE_STORAGE_VALUES start={start:#x} "
                    + " ".join(f"{item:.17g}" for item in values)
                    + "\n"
                )
                position += 1


class SolveEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__(f"*{SOLVE:#x}", internal=False)
        self.count = 0
        self.owner = 0
        self.options = 0
        self.summary = 0

    def stop(self):
        self.count += 1
        gdb.write(
            f"SOLVE_ENTRY count={self.count} options={int(gdb.parse_and_eval('$rdi')):#x} "
            f"data={int(gdb.parse_and_eval('$rsi')):#x} "
            f"solver_options={int(gdb.parse_and_eval('$rdx')):#x}\n"
        )
        if self.count == TARGET_SOLVE:
            self.options = int(gdb.parse_and_eval("$rdi"))
            self.owner = int(gdb.parse_and_eval("$rsi"))
            self.summary = int(gdb.parse_and_eval("$rdx"))
            options = gdb.selected_inferior().read_memory(
                self.options, 488
            ).tobytes()
            fields = (
                ("max_num_iterations", 104, "i"),
                ("num_threads", 120, "i"),
                ("function_tolerance", 184, "d"),
                ("gradient_tolerance", 192, "d"),
                ("parameter_tolerance", 200, "d"),
                ("linear_solver_type", 208, "i"),
                ("preconditioner_type", 212, "i"),
                ("min_linear_solver_iterations", 344, "i"),
                ("max_linear_solver_iterations", 348, "i"),
                ("eta", 352, "d"),
            )
            for name, offset, code in fields:
                value = struct.unpack_from("<" + code, options, offset)[0]
                gdb.write(f"SOLVER_OPTION name={name} value={value}\n")
            dump_reachable_pose_storage(self.owner)
            SolveReturn(self)
        return False


class SolveReturn(gdb.FinishBreakpoint):
    def __init__(self, solve_entry):
        super().__init__(gdb.newest_frame(), internal=True)
        self.solve_entry = solve_entry

    def stop(self):
        gdb.write(f"SOLVE_RETURN count={TARGET_SOLVE}\n")
        inferior = gdb.selected_inferior()
        summary = inferior.read_memory(self.solve_entry.summary, 512).tobytes()
        gdb.write(
            "SOLVER_SUMMARY "
            f"termination={struct.unpack_from('<i', summary, 4)[0]} "
            f"initial_cost={struct.unpack_from('<d', summary, 40)[0]:.17g} "
            f"final_cost={struct.unpack_from('<d', summary, 48)[0]:.17g} "
            f"successful_steps={struct.unpack_from('<i', summary, 88)[0]} "
            f"unsuccessful_steps={struct.unpack_from('<i', summary, 92)[0]}\n"
        )
        iteration_begin, iteration_end = struct.unpack_from("<QQ", summary, 64)
        iteration_stride = 120
        if (
            iteration_end >= iteration_begin
            and (iteration_end - iteration_begin) % iteration_stride == 0
        ):
            iterations = inferior.read_memory(
                iteration_begin, iteration_end - iteration_begin
            ).tobytes()
            for offset in range(0, len(iterations), iteration_stride):
                gdb.write(
                    "SOLVER_ITERATION "
                    f"iteration={struct.unpack_from('<i', iterations, offset)[0]} "
                    f"successful={int(bool(iterations[offset + 6]))} "
                    f"cost={struct.unpack_from('<d', iterations, offset + 8)[0]:.17g} "
                    f"cost_change={struct.unpack_from('<d', iterations, offset + 16)[0]:.17g} "
                    f"gradient_norm={struct.unpack_from('<d', iterations, offset + 32)[0]:.17g} "
                    f"step_norm={struct.unpack_from('<d', iterations, offset + 40)[0]:.17g} "
                    f"relative_decrease={struct.unpack_from('<d', iterations, offset + 48)[0]:.17g} "
                    f"trust_region_radius={struct.unpack_from('<d', iterations, offset + 56)[0]:.17g} "
                    f"linear_iterations={struct.unpack_from('<i', iterations, offset + 92)[0]}\n"
                )
        for submap_index, addresses in enumerate(pose_addresses):
            for address in addresses:
                try:
                    start = address - 96
                    values = struct.unpack(
                        "<32d", inferior.read_memory(start, 32 * 8).tobytes()
                    )
                except gdb.MemoryError:
                    continue
                gdb.write(
                    f"POSE_STORAGE_RESULT submap={submap_index} "
                    f"start={start:#x} "
                    + " ".join(f"{item:.17g}" for item in values)
                    + "\n"
                )
        gdb.execute("set logging enabled off")
        gdb.execute("quit")
        return False


pose_addresses = [[], []]
solve_entry = SolveEntry()
end

continue
