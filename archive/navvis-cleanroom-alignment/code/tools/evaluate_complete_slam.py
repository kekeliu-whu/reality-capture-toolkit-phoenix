#!/usr/bin/env python3
"""End-to-end clean-room SLAM graph and exact 9D IMU evaluation.

The frontend state is the sole source of node/submap initial poses and
topology.  Frozen node/submap archives provide only immutable metadata, while
the reference optimization trajectory is read only after optimization for
gauge alignment and error measurement.  It is never copied into the generated
graph or result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Sequence
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.surveyor_frontend import (  # noqa: E402
    parse_submap_hybrid_grid,
)
from navvis_recon.surveyor_slam import (  # noqa: E402
    ImuCalibration,
    ImuIntrinsics,
    ImuSample,
    LoopConstraint,
    NodeId,
    OptimizationState,
    Rigid3,
    Submap,
    TrajectoryNode,
    build_imu_pose_graph,
    build_pose_graph,
    imu_calibration_from_state,
    load_loop_constraints,
    load_optimization_imu,
    load_optimization_state,
    load_submaps,
    load_trajectory_nodes,
    optimize_imu_pose_graph,
    optimize_imu_pose_graph_ceres,
)


@dataclass(frozen=True, slots=True)
class FrontendState:
    nodes: tuple[TrajectoryNode, ...]
    submaps: tuple[Submap, ...]
    loops: tuple[LoopConstraint, ...]
    schema_version: int | None
    submap_finished_available: bool
    submap_gravity_available: bool
    submap_hybrid_grid_cell_counts: tuple[int, ...] | None
    online_initial_state: OptimizationState | None = None


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")


def _integer(value: float, name: str) -> int:
    integer = int(round(float(value)))
    if not np.isfinite(value) or abs(float(value) - integer) > 1.0e-9:
        raise ValueError(f"{name} must be an integer, got {value}")
    return integer


def load_frontend_state(
    path: Path,
    frozen_nodes: Sequence[TrajectoryNode],
    frozen_submaps: Sequence[Submap],
) -> FrontendState:
    """Rebuild graph inputs from ``evaluate_raw_slam_frontend.py`` NPZ data."""

    required = {
        "node_timestamps_ns",
        "node_translations",
        "node_quaternions_xyzw",
        "submap_start_timestamps_ns",
        "submap_end_timestamps_ns",
        "submap_translations",
        "submap_quaternions_xyzw",
        "submap_membership_offsets",
        "submap_membership_indices",
        "loops",
    }
    optional = {
        "state_schema_version",
        "submap_finished",
        "submap_gravity_observations",
        "submap_hybrid_grid_cell_counts",
        "online_node_translations",
        "online_node_quaternions_xyzw",
        "online_submap_translations",
        "online_submap_quaternions_xyzw",
        "online_gravity_magnitude",
        "online_imu_from_tracking_translation",
        "online_imu_from_tracking_quaternion_xyzw",
        "online_linear_acceleration_bias",
        "online_linear_acceleration_scaling",
        "online_linear_acceleration_cross_axis",
        "online_angular_velocity_bias",
        "online_angular_velocity_scaling",
        "online_angular_velocity_cross_axis",
    }
    with np.load(path, allow_pickle=False) as state:
        missing = sorted(required.difference(state.files))
        if missing:
            raise ValueError(f"frontend state is missing arrays: {', '.join(missing)}")
        arrays = {name: np.asarray(state[name]) for name in required}
        arrays.update(
            {name: np.asarray(state[name]) for name in optional if name in state.files}
        )

    schema_version: int | None = None
    if "state_schema_version" in arrays:
        version_array = arrays["state_schema_version"]
        if version_array.shape not in ((), (1,)):
            raise ValueError("state_schema_version must be a scalar")
        schema_version = _integer(
            float(version_array.reshape(-1)[0]), "state_schema_version"
        )
        if schema_version < 1:
            raise ValueError("state_schema_version must be positive")

    node_timestamps = arrays["node_timestamps_ns"].astype(np.int64, copy=False)
    node_translations = arrays["node_translations"].astype(np.float64, copy=False)
    node_quaternions = arrays["node_quaternions_xyzw"].astype(np.float64, copy=False)
    node_count = len(node_timestamps)
    if node_count < 2:
        raise ValueError("frontend state must contain at least two nodes")
    _require_shape("node_translations", node_translations, (node_count, 3))
    _require_shape("node_quaternions_xyzw", node_quaternions, (node_count, 4))
    if np.any(np.diff(node_timestamps) <= 0):
        raise ValueError("frontend node timestamps must be strictly increasing")

    frozen_node_by_id = {node.node_id: node for node in frozen_nodes}
    nodes: list[TrajectoryNode] = []
    for index in range(node_count):
        node_id = NodeId(0, index)
        metadata = frozen_node_by_id.get(node_id)
        if metadata is None:
            raise ValueError(f"frozen node metadata is missing {node_id}")
        timestamp_ns = int(node_timestamps[index])
        if timestamp_ns != metadata.timestamp_ns:
            raise ValueError(
                f"node {index} timestamp differs from frozen metadata: "
                f"{timestamp_ns} != {metadata.timestamp_ns}"
            )
        generated_pose = Rigid3(node_translations[index], node_quaternions[index])
        nodes.append(
            TrajectoryNode(
                node_id,
                timestamp_ns,
                generated_pose,
                generated_pose,
                metadata.gravity_observation.copy(),
            )
        )

    submap_starts = arrays["submap_start_timestamps_ns"].astype(np.int64, copy=False)
    submap_ends = arrays["submap_end_timestamps_ns"].astype(np.int64, copy=False)
    submap_translations = arrays["submap_translations"].astype(np.float64, copy=False)
    submap_quaternions = arrays["submap_quaternions_xyzw"].astype(
        np.float64, copy=False
    )
    submap_count = len(submap_starts)
    if submap_count < 1:
        raise ValueError("frontend state must contain at least one submap")
    _require_shape("submap_end_timestamps_ns", submap_ends, (submap_count,))
    _require_shape("submap_translations", submap_translations, (submap_count, 3))
    _require_shape(
        "submap_quaternions_xyzw", submap_quaternions, (submap_count, 4)
    )
    if np.any(submap_ends < submap_starts):
        raise ValueError("a frontend submap ends before it starts")

    finished_available = "submap_finished" in arrays
    gravity_available = "submap_gravity_observations" in arrays
    hybrid_grid_counts: tuple[int, ...] | None = None
    if finished_available:
        finished_values = arrays["submap_finished"]
        _require_shape("submap_finished", finished_values, (submap_count,))
        if finished_values.dtype.kind not in "buif" or np.any(
            (finished_values != 0) & (finished_values != 1)
        ):
            raise ValueError("submap_finished must contain only boolean/0/1 values")
        generated_finished = finished_values.astype(np.bool_, copy=False)
    else:
        generated_finished = np.zeros(submap_count, dtype=np.bool_)
    if gravity_available:
        generated_gravity = arrays["submap_gravity_observations"].astype(
            np.float64, copy=False
        )
        _require_shape(
            "submap_gravity_observations", generated_gravity, (submap_count, 3)
        )
    else:
        generated_gravity = np.zeros((submap_count, 3), dtype=np.float64)
    if "submap_hybrid_grid_cell_counts" in arrays:
        cell_counts = arrays["submap_hybrid_grid_cell_counts"]
        _require_shape(
            "submap_hybrid_grid_cell_counts", cell_counts, (submap_count,)
        )
        if cell_counts.dtype.kind not in "iu":
            raise ValueError("submap_hybrid_grid_cell_counts must be integers")
        if np.any(cell_counts < 0):
            raise ValueError("submap_hybrid_grid_cell_counts cannot be negative")
        hybrid_grid_counts = tuple(int(value) for value in cell_counts)

    offsets = arrays["submap_membership_offsets"].astype(np.int64, copy=False)
    indices = arrays["submap_membership_indices"].astype(np.int64, copy=False)
    _require_shape("submap_membership_offsets", offsets, (submap_count + 1,))
    if offsets[0] != 0 or offsets[-1] != len(indices) or np.any(np.diff(offsets) < 0):
        raise ValueError("invalid submap membership offsets")
    if np.any(indices < 0) or np.any(indices >= node_count):
        raise ValueError("submap membership references a missing frontend node")

    frozen_submap_by_id = {submap.submap_id: submap for submap in frozen_submaps}
    submaps: list[Submap] = []
    for index in range(submap_count):
        submap_id = NodeId(0, index)
        metadata = frozen_submap_by_id.get(submap_id)
        if metadata is None:
            raise ValueError(f"frozen submap metadata is missing {submap_id}")
        members = tuple(int(value) for value in indices[offsets[index] : offsets[index + 1]])
        submaps.append(
            Submap(
                submap_id,
                int(submap_starts[index]),
                int(submap_ends[index]),
                Rigid3(submap_translations[index], submap_quaternions[index]),
                members,
                bool(generated_finished[index]),
                generated_gravity[index].copy(),
            )
        )

    loop_rows = arrays["loops"].astype(np.float64, copy=False)
    _require_shape("loops", loop_rows, (len(loop_rows), 13))
    loops: list[LoopConstraint] = []
    for row_index, row in enumerate(loop_rows):
        submap_index = _integer(row[0], f"loops[{row_index}].submap")
        node_index = _integer(row[1], f"loops[{row_index}].node")
        tag = _integer(row[12], f"loops[{row_index}].tag")
        translation_weight = float(row[9])
        rotation_weight = float(row[10])
        valid = bool(row[11] > 0.5)
        if valid and (translation_weight <= 0.0 or rotation_weight <= 0.0):
            raise ValueError("valid loop constraints must have positive weights")
        loops.append(
            LoopConstraint(
                NodeId(0, submap_index),
                NodeId(0, node_index),
                Rigid3(row[2:5], row[5:9]),
                translation_weight,
                rotation_weight,
                valid,
                tag,
            )
        )
    online_names = {
        "online_node_translations",
        "online_node_quaternions_xyzw",
        "online_submap_translations",
        "online_submap_quaternions_xyzw",
        "online_gravity_magnitude",
        "online_imu_from_tracking_translation",
        "online_imu_from_tracking_quaternion_xyzw",
        "online_linear_acceleration_bias",
        "online_linear_acceleration_scaling",
        "online_linear_acceleration_cross_axis",
        "online_angular_velocity_bias",
        "online_angular_velocity_scaling",
        "online_angular_velocity_cross_axis",
    }
    present_online_names = online_names.intersection(arrays)
    if present_online_names and present_online_names != online_names:
        missing_online = sorted(online_names.difference(present_online_names))
        raise ValueError(
            "frontend state has an incomplete online Stage1 state: "
            + ", ".join(missing_online)
        )
    online_initial_state = None
    if present_online_names:
        online_node_translations = arrays["online_node_translations"].astype(
            np.float64, copy=False
        )
        online_node_quaternions = arrays["online_node_quaternions_xyzw"].astype(
            np.float64, copy=False
        )
        online_submap_translations = arrays["online_submap_translations"].astype(
            np.float64, copy=False
        )
        online_submap_quaternions = arrays[
            "online_submap_quaternions_xyzw"
        ].astype(np.float64, copy=False)
        _require_shape(
            "online_node_translations", online_node_translations, (node_count, 3)
        )
        _require_shape(
            "online_node_quaternions_xyzw",
            online_node_quaternions,
            (node_count, 4),
        )
        _require_shape(
            "online_submap_translations",
            online_submap_translations,
            (submap_count, 3),
        )
        _require_shape(
            "online_submap_quaternions_xyzw",
            online_submap_quaternions,
            (submap_count, 4),
        )
        online_shapes = {
            "online_gravity_magnitude": (),
            "online_imu_from_tracking_translation": (3,),
            "online_imu_from_tracking_quaternion_xyzw": (4,),
            "online_linear_acceleration_bias": (3,),
            "online_linear_acceleration_scaling": (3,),
            "online_linear_acceleration_cross_axis": (6,),
            "online_angular_velocity_bias": (3,),
            "online_angular_velocity_scaling": (3,),
            "online_angular_velocity_cross_axis": (6,),
        }
        for name, shape in online_shapes.items():
            arrays[name] = arrays[name].astype(np.float64, copy=False)
            _require_shape(name, arrays[name], shape)
        gravity_magnitude = float(arrays["online_gravity_magnitude"].item())
        if gravity_magnitude <= 0.0:
            raise ValueError("online_gravity_magnitude must be positive")
        online_initial_state = OptimizationState(
            timestamps_ns=node_timestamps.copy(),
            poses=tuple(
                Rigid3(translation, quaternion)
                for translation, quaternion in zip(
                    online_node_translations, online_node_quaternions
                )
            ),
            velocities=np.zeros((node_count, 3), dtype=np.float64),
            submap_poses=tuple(
                Rigid3(translation, quaternion)
                for translation, quaternion in zip(
                    online_submap_translations, online_submap_quaternions
                )
            ),
            gravity_magnitude=gravity_magnitude,
            imu_from_tracking=Rigid3(
                arrays["online_imu_from_tracking_translation"],
                arrays["online_imu_from_tracking_quaternion_xyzw"],
            ),
            linear_acceleration_intrinsics=ImuIntrinsics(
                arrays["online_linear_acceleration_bias"],
                arrays["online_linear_acceleration_scaling"],
                arrays["online_linear_acceleration_cross_axis"],
            ),
            angular_velocity_intrinsics=ImuIntrinsics(
                arrays["online_angular_velocity_bias"],
                arrays["online_angular_velocity_scaling"],
                arrays["online_angular_velocity_cross_axis"],
            ),
        )

    return FrontendState(
        tuple(nodes),
        tuple(submaps),
        tuple(loops),
        schema_version,
        finished_available,
        gravity_available,
        hybrid_grid_counts,
        online_initial_state,
    )


def _valid_loop_pairs(loops: Sequence[LoopConstraint]) -> set[tuple[int, int]]:
    return {
        (loop.submap_id.index, loop.node_id.index)
        for loop in loops
        if loop.valid
    }


def _loops_within_state(
    loops: Sequence[LoopConstraint], state: FrontendState
) -> tuple[LoopConstraint, ...]:
    node_ids = {node.node_id for node in state.nodes}
    submap_ids = {submap.submap_id for submap in state.submaps}
    return tuple(
        loop
        for loop in loops
        if loop.node_id in node_ids and loop.submap_id in submap_ids
    )


def _pair_list(pairs: set[tuple[int, int]]) -> list[list[int]]:
    return [[submap, node] for submap, node in sorted(pairs)]


def loop_report(
    state_loops: Sequence[LoopConstraint],
    frozen_loops: Sequence[LoopConstraint],
    selected_loops: Sequence[LoopConstraint],
    source: str,
) -> dict[str, object]:
    state_pairs = _valid_loop_pairs(state_loops)
    frozen_pairs = _valid_loop_pairs(frozen_loops)
    selected_pairs = _valid_loop_pairs(selected_loops)
    true_positive = state_pairs & frozen_pairs
    false_positive = state_pairs - frozen_pairs
    false_negative = frozen_pairs - state_pairs
    frozen_by_pair = {
        (loop.submap_id.index, loop.node_id.index): loop
        for loop in frozen_loops
        if loop.valid
    }
    measurement_errors = []
    for loop in state_loops:
        pair = (loop.submap_id.index, loop.node_id.index)
        expected = frozen_by_pair.get(pair)
        if not loop.valid or expected is None:
            continue
        measurement_errors.append(
            {
                "submap": pair[0],
                "node": pair[1],
                "translation_m": float(
                    np.linalg.norm(
                        loop.submap_from_node.translation
                        - expected.submap_from_node.translation
                    )
                ),
                "rotation_deg": float(
                    np.degrees(
                        (
                            loop.submap_from_node.rotation.inv()
                            * expected.submap_from_node.rotation
                        ).magnitude()
                    )
                ),
            }
        )
    if measurement_errors:
        translation = np.asarray(
            [value["translation_m"] for value in measurement_errors]
        )
        rotation = np.asarray(
            [value["rotation_deg"] for value in measurement_errors]
        )
        measurement_summary: dict[str, object] | None = {
            "translation_m": _statistics(translation),
            "rotation_deg": _statistics(rotation),
        }
    else:
        measurement_summary = None
    return {
        "source_used_by_graph": source,
        "selected": _pair_list(selected_pairs),
        "state": _pair_list(state_pairs),
        "frozen": _pair_list(frozen_pairs),
        "comparison": {
            "true_positive": len(true_positive),
            "false_positive": len(false_positive),
            "false_negative": len(false_negative),
            "precision": (
                float(len(true_positive) / len(state_pairs)) if state_pairs else None
            ),
            "recall": (
                float(len(true_positive) / len(frozen_pairs)) if frozen_pairs else None
            ),
        },
        "matched_measurement_errors": measurement_errors,
        "matched_measurement_summary": measurement_summary,
    }


def load_frozen_hybrid_grid_cell_counts(
    path: Path, frozen_submaps: Sequence[Submap]
) -> dict[NodeId, int]:
    """Load high-resolution HybridGrid sizes from frozen submap clouds."""

    counts: dict[NodeId, int] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for submap in frozen_submaps:
            name = f"submap_{submap.submap_id.index:08d}.pb"
            if name not in names:
                raise ValueError(f"frozen submap cloud archive is missing {name}")
            grid = parse_submap_hybrid_grid(archive.read(name))
            try:
                counts[submap.submap_id] = grid.cell_count
            finally:
                grid.close()
    return counts


def _vertex_label(kind: str, node_id: NodeId) -> str:
    return f"{kind}:{node_id.trajectory}:{node_id.index}"


def _constraint_graph(
    nodes: Sequence[TrajectoryNode],
    submaps: Sequence[Submap],
    loops: Sequence[LoopConstraint],
) -> tuple[dict[str, object], set[str], set[tuple[str, str]], set[tuple[str, str]]]:
    """Summarize the bipartite node/Submap constraint graph."""

    node_vertices = {_vertex_label("node", node.node_id) for node in nodes}
    submap_vertices = {
        _vertex_label("submap", submap.submap_id) for submap in submaps
    }
    vertices = node_vertices | submap_vertices
    membership_edges: set[tuple[str, str]] = set()
    loop_edges: set[tuple[str, str]] = set()
    dangling: list[dict[str, object]] = []
    adjacency = {vertex: set() for vertex in vertices}

    def add_edge(
        edge_set: set[tuple[str, str]],
        submap_vertex: str,
        node_vertex: str,
        edge_type: str,
    ) -> None:
        if submap_vertex not in vertices or node_vertex not in vertices:
            dangling.append(
                {
                    "type": edge_type,
                    "submap_vertex": submap_vertex,
                    "node_vertex": node_vertex,
                }
            )
            return
        edge_set.add((submap_vertex, node_vertex))
        adjacency[submap_vertex].add(node_vertex)
        adjacency[node_vertex].add(submap_vertex)

    for submap in submaps:
        submap_vertex = _vertex_label("submap", submap.submap_id)
        for node_index in submap.node_indices:
            add_edge(
                membership_edges,
                submap_vertex,
                _vertex_label(
                    "node", NodeId(submap.submap_id.trajectory, node_index)
                ),
                "membership",
            )
    for loop in loops:
        if loop.valid:
            add_edge(
                loop_edges,
                _vertex_label("submap", loop.submap_id),
                _vertex_label("node", loop.node_id),
                "loop",
            )

    components: list[list[str]] = []
    remaining = set(vertices)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: list[str] = []
        remaining.remove(seed)
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in sorted(adjacency[vertex], reverse=True):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda value: (-len(value), value))
    isolated = sorted(vertex for vertex, edges in adjacency.items() if not edges)
    report = {
        "node_vertices": len(node_vertices),
        "submap_vertices": len(submap_vertices),
        "vertices": len(vertices),
        "membership_edges": len(membership_edges),
        "valid_loop_edges": len(loop_edges),
        "unique_graph_edges": len(membership_edges | loop_edges),
        "connected_components": len(components),
        "component_sizes": [len(component) for component in components],
        "isolated_vertices": len(isolated),
        "isolated_vertex_ids": isolated,
        "dangling_edges": len(dangling),
        "dangling_edge_details": dangling,
    }
    return report, vertices, membership_edges, loop_edges


def _hybrid_grid_report(
    state: FrontendState, frozen_counts: dict[NodeId, int]
) -> dict[str, object]:
    generated_counts = state.submap_hybrid_grid_cell_counts
    generated_by_id = (
        {
            submap.submap_id: int(count)
            for submap, count in zip(state.submaps, generated_counts)
        }
        if generated_counts is not None
        else None
    )
    all_ids = sorted(
        set(frozen_counts)
        | (set(generated_by_id) if generated_by_id is not None else set()),
        key=lambda value: (value.trajectory, value.index),
    )
    details = []
    for submap_id in all_ids:
        generated = (
            generated_by_id.get(submap_id) if generated_by_id is not None else None
        )
        frozen = frozen_counts.get(submap_id)
        difference = (
            generated - frozen
            if generated is not None and frozen is not None
            else None
        )
        details.append(
            {
                "trajectory": submap_id.trajectory,
                "submap": submap_id.index,
                "generated_cells": generated,
                "frozen_cells": frozen,
                "difference_cells": difference,
                "absolute_difference_cells": (
                    abs(difference) if difference is not None else None
                ),
                "relative_difference": (
                    difference / frozen
                    if difference is not None and frozen != 0
                    else None
                ),
                "exact": difference == 0 if difference is not None else None,
            }
        )
    available = generated_by_id is not None
    ids_exact = available and set(generated_by_id) == set(frozen_counts)
    common_ids = (
        set(generated_by_id) & set(frozen_counts)
        if generated_by_id is not None
        else set()
    )
    common_generated_total = (
        sum(generated_by_id[value] for value in common_ids)
        if generated_by_id is not None
        else None
    )
    common_frozen_total = (
        sum(frozen_counts[value] for value in common_ids)
        if generated_by_id is not None
        else None
    )
    all_exact = (
        ids_exact
        and all(value["exact"] is True for value in details)
    )
    return {
        "status": "unavailable" if not available else ("exact" if all_exact else "different"),
        "generated_available": available,
        "all_exact": all_exact if available else None,
        "submap_ids_exact": ids_exact if available else None,
        "generated_total_cells": (
            sum(generated_by_id.values()) if generated_by_id is not None else None
        ),
        "frozen_total_cells": sum(frozen_counts.values()),
        "total_difference_cells": (
            sum(generated_by_id.values()) - sum(frozen_counts.values())
            if ids_exact and generated_by_id is not None
            else None
        ),
        "common_submaps": len(common_ids) if generated_by_id is not None else None,
        "common_submap_generated_total_cells": common_generated_total,
        "common_submap_frozen_total_cells": common_frozen_total,
        "common_submap_difference_cells": (
            common_generated_total - common_frozen_total
            if common_generated_total is not None and common_frozen_total is not None
            else None
        ),
        "sum_absolute_difference_cells": (
            sum(
                int(value["absolute_difference_cells"])
                for value in details
                if value["absolute_difference_cells"] is not None
            )
            if generated_by_id is not None
            else None
        ),
        "exact_submaps": (
            sum(value["exact"] is True for value in details)
            if generated_by_id is not None
            else None
        ),
        "details": details,
    }


def topology_report(
    state: FrontendState,
    frozen_nodes: Sequence[TrajectoryNode],
    frozen_submaps: Sequence[Submap],
    frozen_loops: Sequence[LoopConstraint],
    frozen_hybrid_grid_cell_counts: dict[NodeId, int],
) -> dict[str, object]:
    """Strictly compare generated Submap state and graph topology."""

    frozen_by_id = {submap.submap_id: submap for submap in frozen_submaps}
    details = []
    for submap in state.submaps:
        expected = frozen_by_id.get(submap.submap_id)
        gravity_bit_exact = (
            np.array_equal(
                np.asarray(submap.gravity_observation, dtype=np.float64).view(np.uint64),
                np.asarray(expected.gravity_observation, dtype=np.float64).view(np.uint64),
            )
            if state.submap_gravity_available and expected is not None
            else None
        )
        details.append(
            {
                "trajectory": submap.submap_id.trajectory,
                "submap": submap.submap_id.index,
                "present_in_frozen": expected is not None,
                "start_timestamp_exact": (
                    expected is not None
                    and submap.start_timestamp_ns == expected.start_timestamp_ns
                ),
                "end_timestamp_exact": (
                    expected is not None
                    and submap.end_timestamp_ns == expected.end_timestamp_ns
                ),
                "members_exact": (
                    expected is not None
                    and submap.node_indices == expected.node_indices
                ),
                "generated_members": len(submap.node_indices),
                "frozen_members": (
                    len(expected.node_indices) if expected is not None else None
                ),
                "finished_available": state.submap_finished_available,
                "generated_finished": (
                    submap.finished if state.submap_finished_available else None
                ),
                "frozen_finished": expected.finished if expected is not None else None,
                "finished_exact": (
                    submap.finished == expected.finished
                    if state.submap_finished_available and expected is not None
                    else None
                ),
                "gravity_available": state.submap_gravity_available,
                "generated_gravity": (
                    submap.gravity_observation.tolist()
                    if state.submap_gravity_available
                    else None
                ),
                "frozen_gravity": (
                    expected.gravity_observation.tolist()
                    if expected is not None
                    else None
                ),
                "gravity_bit_exact": gravity_bit_exact,
                "gravity_l2_difference": (
                    float(
                        np.linalg.norm(
                            submap.gravity_observation
                            - expected.gravity_observation
                        )
                    )
                    if state.submap_gravity_available and expected is not None
                    else None
                ),
            }
        )

    frozen_in_range = tuple(
        submap
        for submap in frozen_submaps
        if submap.submap_id.trajectory == 0
        and submap.submap_id.index < len(state.submaps)
    )
    generated_ids = {submap.submap_id for submap in state.submaps}
    frozen_ids = {submap.submap_id for submap in frozen_submaps}
    lifecycle_membership_exact = (
        generated_ids == frozen_ids
        and all(
            value["start_timestamp_exact"]
            and value["end_timestamp_exact"]
            and value["members_exact"]
            for value in details
        )
    )
    finished_exact = (
        state.submap_finished_available
        and generated_ids == frozen_ids
        and all(value["finished_exact"] is True for value in details)
    )
    gravity_exact = (
        state.submap_gravity_available
        and generated_ids == frozen_ids
        and all(value["gravity_bit_exact"] is True for value in details)
    )
    hybrid_grid = _hybrid_grid_report(state, frozen_hybrid_grid_cell_counts)

    generated_graph, generated_vertices, generated_memberships, generated_loops = (
        _constraint_graph(state.nodes, state.submaps, state.loops)
    )
    frozen_graph, frozen_vertices, frozen_memberships, frozen_loop_edges = (
        _constraint_graph(frozen_nodes, frozen_submaps, frozen_loops)
    )
    graph_comparison = {
        "vertices_exact": generated_vertices == frozen_vertices,
        "membership_edges_exact": generated_memberships == frozen_memberships,
        "loop_edges_exact": generated_loops == frozen_loop_edges,
        "connected_components_exact": (
            generated_graph["connected_components"]
            == frozen_graph["connected_components"]
            and generated_graph["component_sizes"] == frozen_graph["component_sizes"]
        ),
        "isolated_vertices_exact": (
            generated_graph["isolated_vertex_ids"]
            == frozen_graph["isolated_vertex_ids"]
        ),
    }
    graph_comparison["all_exact"] = all(graph_comparison.values())

    unavailable = []
    if not state.submap_finished_available:
        unavailable.append("submap_finished")
    if not state.submap_gravity_available:
        unavailable.append("submap_gravity_observations")
    if state.submap_hybrid_grid_cell_counts is None:
        unavailable.append("submap_hybrid_grid_cell_counts")
    strict_all_exact = (
        not unavailable
        and lifecycle_membership_exact
        and finished_exact
        and gravity_exact
        and hybrid_grid["all_exact"] is True
        and graph_comparison["all_exact"] is True
    )
    return {
        "status": "unavailable" if unavailable else ("exact" if strict_all_exact else "different"),
        "all_exact": strict_all_exact,
        "strict_all_exact": strict_all_exact,
        "unavailable_generated_fields": unavailable,
        "state_schema_version": state.schema_version,
        "generated_submap_ids_exact": generated_ids == frozen_ids,
        "lifecycle_membership_all_exact": lifecycle_membership_exact,
        "finished_all_exact": finished_exact if state.submap_finished_available else None,
        "gravity_all_bit_exact": gravity_exact if state.submap_gravity_available else None,
        "generated_submaps": len(state.submaps),
        "frozen_submaps": len(frozen_submaps),
        "frozen_submaps_in_state_range": len(frozen_in_range),
        "generated_memberships": sum(
            len(submap.node_indices) for submap in state.submaps
        ),
        "frozen_memberships": sum(
            len(submap.node_indices) for submap in frozen_submaps
        ),
        "frozen_memberships_in_state_range": sum(
            len(submap.node_indices) for submap in frozen_in_range
        ),
        "missing_generated_submap_ids": [
            [value.trajectory, value.index]
            for value in sorted(
                frozen_ids - generated_ids,
                key=lambda item: (item.trajectory, item.index),
            )
        ],
        "extra_generated_submap_ids": [
            [value.trajectory, value.index]
            for value in sorted(
                generated_ids - frozen_ids,
                key=lambda item: (item.trajectory, item.index),
            )
        ],
        "hybrid_grid": hybrid_grid,
        "constraint_graph": {
            "definition": (
                "bipartite trajectory-node/Submap graph with membership and "
                "valid loop edges"
            ),
            "generated": generated_graph,
            "frozen": frozen_graph,
            "comparison": graph_comparison,
        },
        "details": details,
    }


def gauge_align(
    estimate: Sequence[Rigid3], reference: Sequence[Rigid3]
) -> tuple[tuple[Rigid3, ...], Rigid3]:
    if not estimate or len(estimate) != len(reference):
        raise ValueError("estimate and reference pose sequences must have equal size")
    reference_from_estimate = reference[0].compose(estimate[0].inverse())
    return (
        tuple(reference_from_estimate.compose(pose) for pose in estimate),
        reference_from_estimate,
    )


def _statistics(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot summarize an empty metric")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def ate_errors(
    estimate: Sequence[Rigid3], reference: Sequence[Rigid3]
) -> dict[str, object]:
    translation = np.asarray(
        [
            np.linalg.norm(actual.translation - expected.translation)
            for actual, expected in zip(estimate, reference)
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [
            np.degrees((actual.rotation.inv() * expected.rotation).magnitude())
            for actual, expected in zip(estimate, reference)
        ],
        dtype=np.float64,
    )
    return {
        "pose_count": len(translation),
        "translation_m": _statistics(translation),
        "rotation_deg": _statistics(rotation),
    }


def rpe_errors(
    estimate: Sequence[Rigid3],
    reference: Sequence[Rigid3],
    timestamps_ns: np.ndarray,
    *,
    interval_seconds: float,
    tolerance_ms: float,
) -> dict[str, object]:
    interval_ns = int(round(interval_seconds * 1.0e9))
    tolerance_ns = int(round(tolerance_ms * 1.0e6))
    translation: list[float] = []
    rotation: list[float] = []
    for first, timestamp in enumerate(timestamps_ns):
        second = int(np.searchsorted(timestamps_ns, int(timestamp) + interval_ns))
        if second >= len(timestamps_ns):
            continue
        if abs(int(timestamps_ns[second]) - int(timestamp) - interval_ns) > tolerance_ns:
            continue
        estimate_relative = estimate[first].between(estimate[second])
        reference_relative = reference[first].between(reference[second])
        error = reference_relative.inverse().compose(estimate_relative)
        translation.append(float(np.linalg.norm(error.translation)))
        rotation.append(float(np.degrees(error.rotation.magnitude())))
    if not translation:
        raise ValueError(
            "no timestamp pairs satisfy the requested RPE interval and tolerance"
        )
    return {
        "interval_seconds": interval_seconds,
        "tolerance_ms": tolerance_ms,
        "pair_count": len(translation),
        "translation_m": _statistics(np.asarray(translation)),
        "rotation_deg": _statistics(np.asarray(rotation)),
    }


def _reference_poses_for_timestamps(
    reference: OptimizationState, timestamps_ns: Sequence[int]
) -> tuple[Rigid3, ...]:
    pose_by_timestamp = {
        int(timestamp): pose
        for timestamp, pose in zip(reference.timestamps_ns, reference.poses)
    }
    missing = [
        int(timestamp)
        for timestamp in timestamps_ns
        if int(timestamp) not in pose_by_timestamp
    ]
    if missing:
        raise ValueError(
            f"reference optimization state is missing {len(missing)} node timestamps"
        )
    return tuple(pose_by_timestamp[int(timestamp)] for timestamp in timestamps_ns)


def _default_paths(dataset: Path) -> dict[str, Path]:
    return {
        "nodes": dataset
        / "internal/nodes/trajectory_node/trajectory_node_00000000.zip",
        "submaps": dataset / "internal/submaps/submap/submap_00000000.zip",
        "submap_clouds": dataset
        / "internal/submaps/submap_clouds/submap_clouds_00000000.zip",
        "loops": dataset
        / "internal/constraints_inter_dataset"
        / dataset.name
        / "constraints/constraint_data/constraint_data_00000000.zip",
        "optimization_data": dataset / "artifacts/optimization_data.pb",
        "initial_state": dataset / "artifacts/optimization_state.pb",
        "reference_state": dataset
        / "internal/anchors/optimization/optimization_state.pb",
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    total_start = time.perf_counter()
    dataset = args.dataset.resolve()
    defaults = _default_paths(dataset)
    paths = {
        "nodes": (args.nodes or defaults["nodes"]).resolve(),
        "submaps": (args.submaps or defaults["submaps"]).resolve(),
        "submap_clouds": (
            args.submap_clouds or defaults["submap_clouds"]
        ).resolve(),
        "loops": (args.loops or defaults["loops"]).resolve(),
        "optimization_data": (
            args.optimization_data or defaults["optimization_data"]
        ).resolve(),
        "initial_state": (
            args.initial_state or defaults["initial_state"]
        ).resolve(),
        "reference_state": (
            args.reference_state or defaults["reference_state"]
        ).resolve(),
    }
    if args.state is not None:
        paths["state"] = args.state.resolve()
    # A schema-v3 frontend state contains the generated online Stage1 result,
    # so the installed pre-intrinsics state is only a legacy/isolation fallback.
    required_paths = {
        name: path for name, path in paths.items() if name != "initial_state"
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing evaluation inputs: " + ", ".join(missing))

    frozen_start = time.perf_counter()
    frozen_nodes = load_trajectory_nodes(paths["nodes"])
    frozen_submaps = load_submaps(paths["submaps"])
    frozen_hybrid_grid_cell_counts = load_frozen_hybrid_grid_cell_counts(
        paths["submap_clouds"], frozen_submaps
    )
    frozen_loops_all = load_loop_constraints(paths["loops"])
    samples = load_optimization_imu(paths["optimization_data"])
    frozen_seconds = time.perf_counter() - frozen_start

    state_start = time.perf_counter()
    if args.frozen_stage2_isolation:
        state = FrontendState(
            tuple(frozen_nodes),
            tuple(frozen_submaps),
            tuple(frozen_loops_all),
            None,
            True,
            True,
            tuple(
                frozen_hybrid_grid_cell_counts[submap.submap_id]
                for submap in frozen_submaps
            ),
        )
    else:
        state = load_frontend_state(paths["state"], frozen_nodes, frozen_submaps)
    if state.online_initial_state is not None and not args.frozen_stage2_isolation:
        stage2_initial = state.online_initial_state
        stage2_initial_source = "generated_online_stage1_frontend_npz"
        official_initial_state_loaded = False
    else:
        if not paths["initial_state"].is_file():
            raise FileNotFoundError(
                "legacy frontend state requires the pre-intrinsics initial state: "
                + str(paths["initial_state"])
            )
        stage2_initial = load_optimization_state(paths["initial_state"])
        stage2_initial_source = (
            "frozen_pre_intrinsics_state_for_backend_isolation"
            if args.frozen_stage2_isolation
            else "official_pre_intrinsics_state_legacy_fallback"
        )
        official_initial_state_loaded = True
    frozen_loops = _loops_within_state(frozen_loops_all, state)
    state_seconds = time.perf_counter() - state_start

    if args.frozen_stage2_isolation:
        selected_loops = frozen_loops
        selected_loop_source = "frozen_stage2_backend_isolation"
    elif args.loop_source == "state":
        selected_loops = state.loops
        selected_loop_source = "state"
    else:
        selected_loops = frozen_loops
        selected_loop_source = "frozen"

    graph_start = time.perf_counter()
    pose_graph = build_pose_graph(state.nodes, state.submaps, selected_loops)
    pose_graph_seconds = time.perf_counter() - graph_start

    imu_start = time.perf_counter()
    imu_problem = build_imu_pose_graph(
        pose_graph,
        state.nodes,
        samples,
        imu_calibration_from_state(stage2_initial),
        initial_state=stage2_initial,
    )
    imu_graph_seconds = time.perf_counter() - imu_start

    optimization_start = time.perf_counter()
    if args.ceres_solver is not None:
        if args.fixed_imu_calibration:
            raise ValueError(
                "the native Stage2 worker implements joint calibration; "
                "--fixed-imu-calibration is incompatible with --ceres-solver"
            )
        if args.ceres_work_dir is None:
            raise ValueError("--ceres-work-dir is required with --ceres-solver")
        result = optimize_imu_pose_graph_ceres(
            imu_problem,
            args.ceres_solver,
            args.ceres_work_dir,
            max_iterations=args.max_iterations,
            num_threads=args.solver_threads,
        )
        solver_backend = "native_cpp_ceres_exact_9d_joint_calibration"
    else:
        result = optimize_imu_pose_graph(
            imu_problem,
            max_iterations=args.max_iterations,
            calibrate_imu_intrinsics=not args.fixed_imu_calibration,
        )
        solver_backend = "python_scipy_exact_9d_joint_calibration"
    optimization_seconds = time.perf_counter() - optimization_start

    # The final official state is deliberately deserialized only after the
    # clean solve has completed. It can therefore affect metrics, never the
    # native problem schema, initialization or optimization trajectory.
    reference_load_start = time.perf_counter()
    reference = load_optimization_state(paths["reference_state"])
    reference_load_seconds = time.perf_counter() - reference_load_start

    metric_start = time.perf_counter()
    node_count = len(state.nodes)
    submap_count = len(state.submaps)
    node_estimate = result.poses[:node_count]
    submap_estimate = result.poses[node_count : node_count + submap_count]
    reference_nodes = _reference_poses_for_timestamps(
        reference, [node.timestamp_ns for node in state.nodes]
    )
    if len(reference.submap_poses) < submap_count:
        raise ValueError("reference optimization state has too few submap poses")
    reference_submaps = reference.submap_poses[:submap_count]
    frontend_nodes = tuple(node.local_pose for node in state.nodes)
    frontend_submaps = tuple(submap.local_pose for submap in state.submaps)
    aligned_frontend_nodes, frontend_gauge = gauge_align(
        frontend_nodes, reference_nodes
    )
    aligned_frontend_submaps = tuple(
        frontend_gauge.compose(pose) for pose in frontend_submaps
    )
    aligned_nodes, gauge = gauge_align(node_estimate, reference_nodes)
    aligned_submaps = tuple(gauge.compose(pose) for pose in submap_estimate)
    metrics = {
        "frontend_gauge_transform": {
            "translation_m": frontend_gauge.translation.tolist(),
            "quaternion_xyzw": frontend_gauge.quaternion_xyzw.tolist(),
        },
        "frontend_gauge_aligned_ate_nodes": ate_errors(
            aligned_frontend_nodes, reference_nodes
        ),
        "frontend_gauge_aligned_ate_submaps": ate_errors(
            aligned_frontend_submaps, reference_submaps
        ),
        "gauge_transform": {
            "translation_m": gauge.translation.tolist(),
            "quaternion_xyzw": gauge.quaternion_xyzw.tolist(),
        },
        "gauge_aligned_ate_nodes": ate_errors(aligned_nodes, reference_nodes),
        "gauge_aligned_ate_submaps": ate_errors(
            aligned_submaps, reference_submaps
        ),
        "gauge_aligned_rpe_nodes": rpe_errors(
            aligned_nodes,
            reference_nodes,
            np.asarray([node.timestamp_ns for node in state.nodes], dtype=np.int64),
            interval_seconds=args.rpe_seconds,
            tolerance_ms=args.rpe_tolerance_ms,
        ),
    }
    result_calibration = result.calibration or imu_problem.calibration
    reference_calibration = imu_calibration_from_state(reference)
    calibration_metrics = {
        "estimated": {
            "gravity_magnitude": result_calibration.gravity_magnitude,
            "imu_orientation_xyzw": (
                result_calibration.imu_from_tracking.quaternion_xyzw.tolist()
            ),
            "linear_acceleration_bias": (
                result_calibration.linear_acceleration_bias.tolist()
            ),
            "linear_acceleration_scaling": (
                result_calibration.linear_acceleration_scaling.tolist()
            ),
            "angular_velocity_bias": (
                result_calibration.angular_velocity_bias.tolist()
            ),
            "angular_velocity_scaling": (
                result_calibration.angular_velocity_scaling.tolist()
            ),
        },
        "reference": {
            "gravity_magnitude": reference_calibration.gravity_magnitude,
            "imu_orientation_xyzw": (
                reference_calibration.imu_from_tracking.quaternion_xyzw.tolist()
            ),
            "linear_acceleration_bias": (
                reference_calibration.linear_acceleration_bias.tolist()
            ),
            "linear_acceleration_scaling": (
                reference_calibration.linear_acceleration_scaling.tolist()
            ),
            "angular_velocity_bias": (
                reference_calibration.angular_velocity_bias.tolist()
            ),
            "angular_velocity_scaling": (
                reference_calibration.angular_velocity_scaling.tolist()
            ),
        },
        "absolute_error": {
            "gravity_magnitude": abs(
                result_calibration.gravity_magnitude
                - reference_calibration.gravity_magnitude
            ),
            "imu_orientation_deg": float(
                np.degrees(
                    (
                        reference_calibration.imu_from_tracking.rotation.inv()
                        * result_calibration.imu_from_tracking.rotation
                    ).magnitude()
                )
            ),
            "linear_acceleration_bias_max": float(
                np.max(
                    np.abs(
                        result_calibration.linear_acceleration_bias
                        - reference_calibration.linear_acceleration_bias
                    )
                )
            ),
            "linear_acceleration_scaling_max": float(
                np.max(
                    np.abs(
                        result_calibration.linear_acceleration_scaling
                        - reference_calibration.linear_acceleration_scaling
                    )
                )
            ),
            "angular_velocity_bias_max": float(
                np.max(
                    np.abs(
                        result_calibration.angular_velocity_bias
                        - reference_calibration.angular_velocity_bias
                    )
                )
            ),
            "angular_velocity_scaling_max": float(
                np.max(
                    np.abs(
                        result_calibration.angular_velocity_scaling
                        - reference_calibration.angular_velocity_scaling
                    )
                )
            ),
        },
    }
    metric_seconds = time.perf_counter() - metric_start

    payload: dict[str, object] = {
        "scope": (
            "stage2_backend_isolation_using_frozen_pre_intrinsics_state"
            if args.frozen_stage2_isolation
            else "complete_generated_frontend_and_stage2_backend"
        ),
        "invocation": {
            "argv": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        },
        "inputs": {name: str(path) for name, path in paths.items()},
        "provenance": {
            "graph_pose_and_topology_source": (
                "frozen_archives_with_pre_intrinsics_stage1_state"
                if args.frozen_stage2_isolation
                else "frontend_npz_state"
            ),
            "generated_submap_finished_source": (
                "frozen_submap_archive_for_backend_isolation"
                if args.frozen_stage2_isolation
                else "frontend_npz_state"
                if state.submap_finished_available
                else "unavailable_in_legacy_frontend_npz"
            ),
            "generated_submap_gravity_source": (
                "frozen_submap_archive_for_backend_isolation"
                if args.frozen_stage2_isolation
                else "frontend_npz_state"
                if state.submap_gravity_available
                else "unavailable_in_legacy_frontend_npz"
            ),
            "generated_hybrid_grid_cell_count_source": (
                "frozen_submap_cloud_archive_for_backend_isolation"
                if args.frozen_stage2_isolation
                else "frontend_npz_state"
                if state.submap_hybrid_grid_cell_counts is not None
                else "unavailable_in_legacy_frontend_npz"
            ),
            "frozen_hybrid_grid_cell_count_source": "frozen_submap_cloud_archive",
            "legacy_state_fields_copied_from_frozen": False,
            "frozen_archives_used_as_solver_topology": args.frozen_stage2_isolation,
            "node_gravity_source": "frozen_node_metadata_not_used_by_pose_graph",
            "official_navvis_binary_executed": False,
            "reference_trajectory_usage": "post-optimization metrics only",
            "reference_state_loaded_after_clean_solve": True,
            "stage2_initial_state_source": stage2_initial_source,
            "official_pre_intrinsics_state_loaded": official_initial_state_loaded,
            "imu_calibration_source": (
                stage2_initial_source + "_plus_live_joint_calibration"
                if not args.fixed_imu_calibration
                else stage2_initial_source + "_fixed"
            ),
            "reference_trajectory_copied_into_result": False,
        },
        "counts": {
            "nodes": node_count,
            "submaps": submap_count,
            "memberships": sum(len(submap.node_indices) for submap in state.submaps),
            "state_loops_valid": len(_valid_loop_pairs(state.loops)),
            "frozen_loops_valid_in_state_range": len(_valid_loop_pairs(frozen_loops)),
            "selected_loops_valid": len(_valid_loop_pairs(selected_loops)),
            "pose_graph_edges": len(pose_graph.edges),
            "imu_samples": len(samples),
            "imu_factors": len(imu_problem.preintegrations),
        },
        "loops": loop_report(
            state.loops, frozen_loops, selected_loops, selected_loop_source
        ),
        "topology": topology_report(
            state,
            frozen_nodes,
            frozen_submaps,
            frozen_loops_all,
            frozen_hybrid_grid_cell_counts,
        ),
        "solver": {
            "backend": solver_backend,
            "live_imu_calibration": not args.fixed_imu_calibration,
            "binary": (
                {
                    "path": str(args.ceres_solver.resolve()),
                    "sha256": hashlib.sha256(
                        args.ceres_solver.resolve().read_bytes()
                    ).hexdigest(),
                    "threads": args.solver_threads,
                }
                if args.ceres_solver is not None
                else None
            ),
            "max_iterations": args.max_iterations,
            "iterations": result.iterations,
            "success": result.success,
            "message": result.message,
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
        },
        "imu_calibration": calibration_metrics,
        "metrics": metrics,
        "timing_seconds": {
            "load_frozen_inputs": frozen_seconds,
            "load_frontend_state": state_seconds,
            "build_pose_graph": pose_graph_seconds,
            "build_imu_graph_and_preintegrate": imu_graph_seconds,
            "optimize_exact_9d_imu": optimization_seconds,
            "load_reference_state_after_optimization": reference_load_seconds,
            "evaluate_metrics": metric_seconds,
            "total": time.perf_counter() - total_start,
        },
    }
    return payload


def run_smoke_test() -> dict[str, object]:
    """Exercise NPZ loading, graph construction, 9D backend and metrics."""

    timestamps = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    translations = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    metadata_nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            int(timestamps[index]),
            Rigid3(translations[index], quaternions[index]),
            Rigid3(translations[index], quaternions[index]),
            np.array([0.0, 0.0, 1.0]),
        )
        for index in range(3)
    )
    identity = Rigid3(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    metadata_submaps = (
        Submap(
            NodeId(0, 0),
            0,
            int(timestamps[-1]),
            identity,
            (0, 1, 2),
            True,
            np.array([0.0, 0.0, 1.0]),
        ),
    )
    loop_row = np.array(
        [[0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 500.0, 1600.0, 1.0, 1.0]]
    )
    with tempfile.TemporaryDirectory(prefix="complete_slam_smoke_") as directory:
        state_path = Path(directory) / "state.npz"
        np.savez_compressed(
            state_path,
            state_schema_version=np.asarray(2, dtype=np.int64),
            node_timestamps_ns=timestamps,
            node_translations=translations,
            node_quaternions_xyzw=quaternions,
            submap_start_timestamps_ns=np.array([0], dtype=np.int64),
            submap_end_timestamps_ns=np.array([timestamps[-1]], dtype=np.int64),
            submap_translations=np.zeros((1, 3)),
            submap_quaternions_xyzw=quaternions[:1],
            submap_membership_offsets=np.array([0, 3], dtype=np.int64),
            submap_membership_indices=np.array([0, 1, 2], dtype=np.int64),
            submap_finished=np.array([True], dtype=np.bool_),
            submap_gravity_observations=np.array([[0.0, 0.0, 1.0]]),
            submap_hybrid_grid_cell_counts=np.array([2], dtype=np.int64),
            loops=loop_row,
        )
        state = load_frontend_state(state_path, metadata_nodes, metadata_submaps)

    samples = tuple(
        ImuSample(
            int(timestamp),
            np.array([0.0, 0.0, 9.80665]),
            np.zeros(3),
        )
        for timestamp in np.arange(0, 2_000_000_001, 500_000_000, dtype=np.int64)
    )
    graph = build_pose_graph(state.nodes, state.submaps, state.loops)
    problem = build_imu_pose_graph(
        graph, state.nodes, samples, ImuCalibration(), initial_state=None
    )
    start = time.perf_counter()
    result = optimize_imu_pose_graph(problem, max_iterations=3)
    elapsed = time.perf_counter() - start
    aligned, _ = gauge_align(result.poses[:3], tuple(node.local_pose for node in state.nodes))
    ate = ate_errors(aligned, tuple(node.local_pose for node in state.nodes))
    rpe = rpe_errors(
        aligned,
        tuple(node.local_pose for node in state.nodes),
        timestamps,
        interval_seconds=1.0,
        tolerance_ms=1.0,
    )
    if not result.success:
        raise RuntimeError(f"synthetic exact-backend solve failed: {result.message}")
    return {
        "smoke_test": "passed",
        "coverage": [
            "frontend NPZ reader",
            "generated Submap finished/gravity/HybridGrid state",
            "clean-room pose graph",
            "exact 9D IMU backend",
            "gauge-aligned ATE/RPE",
        ],
        "counts": {
            "nodes": len(state.nodes),
            "submaps": len(state.submaps),
            "loops": len(state.loops),
            "imu_factors": len(problem.preintegrations),
        },
        "solver": {
            "success": result.success,
            "iterations": result.iterations,
            "elapsed_seconds": elapsed,
        },
        "ate": ate,
        "rpe": rpe,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a generated frontend NPZ with the clean-room graph and "
            "existing exact 9D IMU backend against frozen NavVis artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  evaluate_complete_slam.py DATASET --state frontend_state.npz "
            "--output complete_slam.json\n\n"
            "Use --loop-source frozen only for a backend-isolation regression; "
            "the default state mode evaluates generated loop constraints."
        ),
    )
    parser.add_argument("dataset", nargs="?", type=Path, help="Frozen dataset root")
    parser.add_argument(
        "--state",
        type=Path,
        help="NPZ emitted by evaluate_raw_slam_frontend.py --state-output",
    )
    parser.add_argument(
        "--frozen-stage2-isolation",
        action="store_true",
        help=(
            "Use frozen node/Submap/loop topology plus the pre-intrinsics Stage1 "
            "state; no generated frontend NPZ is required"
        ),
    )
    parser.add_argument(
        "--loop-source",
        choices=("state", "frozen"),
        default="state",
        help="Loop measurements used by the generated graph",
    )
    parser.add_argument("--nodes", type=Path, help="Override frozen node archive")
    parser.add_argument("--submaps", type=Path, help="Override frozen submap archive")
    parser.add_argument(
        "--submap-clouds", type=Path, help="Override frozen submap cloud archive"
    )
    parser.add_argument("--loops", type=Path, help="Override frozen loop archive")
    parser.add_argument(
        "--optimization-data", type=Path, help="Override frozen optimization_data.pb"
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        help="Override the pre-intrinsics optimization_state.pb",
    )
    parser.add_argument(
        "--reference-state",
        type=Path,
        help="Override final reference optimization_state.pb",
    )
    parser.add_argument(
        "--fixed-imu-calibration",
        action="store_true",
        help="Disable live shared IMU calibration for backend isolation",
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument(
        "--ceres-solver",
        type=Path,
        help="Use the native C++/Ceres Stage2 worker instead of SciPy",
    )
    parser.add_argument(
        "--ceres-work-dir",
        type=Path,
        help="Directory for versioned native Stage2 problem/result files",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=7,
        help="Ceres worker thread count",
    )
    parser.add_argument("--rpe-seconds", type=float, default=1.0)
    parser.add_argument("--rpe-tolerance-ms", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a self-contained three-node end-to-end smoke test",
    )
    return parser


def main() -> int:
    parser = argument_parser()
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.solver_threads < 1:
        parser.error("--solver-threads must be positive")
    if args.ceres_solver is not None and args.ceres_work_dir is None:
        parser.error("--ceres-work-dir is required with --ceres-solver")
    if args.rpe_seconds <= 0.0 or args.rpe_tolerance_ms < 0.0:
        parser.error("RPE interval must be positive and tolerance non-negative")
    if args.smoke_test:
        payload = run_smoke_test()
    else:
        if args.dataset is None or (
            args.state is None and not args.frozen_stage2_isolation
        ):
            parser.error(
                "dataset and --state are required unless --smoke-test or "
                "--frozen-stage2-isolation is used"
            )
        payload = evaluate(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
