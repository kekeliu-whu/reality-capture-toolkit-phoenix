"""Production handoff from the native frontend to the clean-room SLAM backends.

The binary state accepted here is emitted by ``navvis_recon_slam
--state-output``.  It contains only generated frontend data.  Frozen NavVis
artifacts are deliberately not part of this interchange format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np

from navvis_recon.surveyor_frontend import (
    FrontendNode,
    FrontendSubmap,
    HybridProbabilityGrid,
    _updated_submap_gravity,
    compute_rotational_histogram,
)
from navvis_recon.surveyor_slam import NodeId, Rigid3, Submap, TrajectoryNode


STATE_MAGIC = b"NVCRSLAMSTATE01\0"
STATE_VERSION = 1
ENDIAN_MARKER = 0x01020304
MAXIMUM_STATE_ITEMS = 100_000_000


@dataclass(frozen=True, slots=True)
class NativeFrontendState:
    """Generated local frontend data in loop and backend representations."""

    frontend_nodes: tuple[FrontendNode, ...]
    frontend_submaps: tuple[FrontendSubmap, ...]
    backend_nodes: tuple[TrajectoryNode, ...]
    backend_submaps: tuple[Submap, ...]


def _read_array(
    stream: BinaryIO, dtype: str | np.dtype, count: int, name: str
) -> np.ndarray:
    if count < 0 or count > MAXIMUM_STATE_ITEMS:
        raise ValueError(f"invalid {name} count: {count}")
    values = np.fromfile(stream, dtype=np.dtype(dtype), count=count)
    if len(values) != count:
        raise ValueError(f"truncated native frontend state while reading {name}")
    return values


def _read_scalar(stream: BinaryIO, dtype: str, name: str) -> int:
    return int(_read_array(stream, dtype, 1, name)[0])


def _pose(values: np.ndarray) -> Rigid3:
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("native frontend state contains an invalid pose")
    return Rigid3(values[:3], values[3:])


def _submap_gravity(
    submap_pose: Rigid3,
    members: Sequence[int],
    nodes: Sequence[FrontendNode],
) -> tuple[np.ndarray, int]:
    state = np.zeros(3, dtype=np.float64)
    count = 0
    for index in members:
        if index < 0 or index >= len(nodes):
            raise ValueError(f"Submap membership references missing node {index}")
        node = nodes[index]
        if node.gravity_observation is None:
            continue
        state = _updated_submap_gravity(
            state,
            count,
            submap_pose,
            node.local_pose,
            node.gravity_observation,
        )
        count += 1
    return state, count


def load_native_frontend_state(
    path: str | Path,
    *,
    compute_loop_histograms: bool = True,
    loop_node_stride: int = 10,
    loop_histogram_size: int = 120,
) -> NativeFrontendState:
    """Load a generated native frontend state without reference metadata."""

    if loop_node_stride < 1 or loop_histogram_size < 1:
        raise ValueError("loop histogram parameters must be positive")
    source = Path(path).resolve()
    with source.open("rb") as stream:
        magic = stream.read(len(STATE_MAGIC))
        if magic != STATE_MAGIC:
            raise ValueError(f"invalid native frontend state magic: {source}")
        version = _read_scalar(stream, "<u4", "state version")
        endian = _read_scalar(stream, "<u4", "endian marker")
        if version != STATE_VERSION:
            raise ValueError(f"unsupported native frontend state version {version}")
        if endian != ENDIAN_MARKER:
            raise ValueError("native frontend state endian marker mismatch")
        node_count = _read_scalar(stream, "<u8", "node count")
        submap_count = _read_scalar(stream, "<u8", "Submap count")
        if node_count < 2 or submap_count < 1:
            raise ValueError("native frontend state has incomplete topology")

        frontend_nodes: list[FrontendNode] = []
        backend_nodes: list[TrajectoryNode] = []
        previous_timestamp = -1
        empty_normals = np.empty((0, 3), dtype=np.float32)
        for index in range(node_count):
            timestamp_ns = _read_scalar(stream, "<i8", f"node {index} timestamp")
            if timestamp_ns <= previous_timestamp:
                raise ValueError("native frontend timestamps are not increasing")
            previous_timestamp = timestamp_ns
            pose = _pose(_read_array(stream, "<f8", 7, f"node {index} pose"))
            gravity = _read_array(stream, "<f8", 3, f"node {index} gravity")
            if not np.all(np.isfinite(gravity)):
                raise ValueError(f"node {index} has invalid gravity")
            point_count = _read_scalar(
                stream, "<u8", f"node {index} point count"
            )
            points = _read_array(
                stream, "<f4", 3 * point_count, f"node {index} points"
            ).reshape((-1, 3))
            histogram = None
            if compute_loop_histograms and index % loop_node_stride == 0:
                histogram = compute_rotational_histogram(
                    points, loop_histogram_size
                )
            node_id = NodeId(0, index)
            frontend_nodes.append(
                FrontendNode(
                    node_id,
                    timestamp_ns,
                    pose,
                    points,
                    empty_normals,
                    None,
                    gravity.copy(),
                    histogram,
                )
            )
            backend_nodes.append(
                TrajectoryNode(
                    node_id,
                    timestamp_ns,
                    pose,
                    pose,
                    gravity.copy(),
                )
            )

        frontend_submaps: list[FrontendSubmap] = []
        backend_submaps: list[Submap] = []
        for position in range(submap_count):
            index = _read_scalar(stream, "<u8", f"Submap {position} index")
            if index != position:
                raise ValueError(
                    f"native frontend Submap order mismatch: {index} != {position}"
                )
            start_ns = _read_scalar(stream, "<i8", f"Submap {index} start")
            end_ns = _read_scalar(stream, "<i8", f"Submap {index} end")
            pose = _pose(
                _read_array(stream, "<f8", 7, f"Submap {index} pose")
            )
            finished = _read_scalar(stream, "u1", f"Submap {index} finished")
            if finished not in (0, 1):
                raise ValueError(f"Submap {index} has invalid finished flag")
            member_count = _read_scalar(
                stream, "<u8", f"Submap {index} membership count"
            )
            members = _read_array(
                stream, "<u8", member_count, f"Submap {index} memberships"
            ).astype(np.int64, copy=False)
            if len(members) and (
                np.any(members >= node_count) or np.any(np.diff(members) <= 0)
            ):
                raise ValueError(f"Submap {index} has invalid memberships")

            levels: list[tuple[np.ndarray, np.ndarray]] = []
            for level in range(3):
                count = _read_scalar(
                    stream, "<u8", f"Submap {index} level {level} count"
                )
                points = _read_array(
                    stream,
                    "<f4",
                    3 * count,
                    f"Submap {index} level {level} points",
                ).reshape((-1, 3))
                normals = _read_array(
                    stream,
                    "<f4",
                    3 * count,
                    f"Submap {index} level {level} normals",
                ).reshape((-1, 3))
                levels.append((points, normals))

            cell_count = _read_scalar(
                stream, "<u8", f"Submap {index} HybridGrid count"
            )
            grid_indices = _read_array(
                stream,
                "<i4",
                3 * cell_count,
                f"Submap {index} HybridGrid indices",
            ).reshape((-1, 3))
            grid_values = _read_array(
                stream,
                "<u2",
                cell_count,
                f"Submap {index} HybridGrid values",
            )
            grid = HybridProbabilityGrid(
                indices=grid_indices, values=grid_values
            )
            member_list = [int(value) for value in members]
            gravity, gravity_count = _submap_gravity(
                pose, member_list, frontend_nodes
            )
            submap_id = NodeId(0, index)
            frontend_submap = FrontendSubmap(
                submap_id,
                pose,
                start_ns,
                0.0,
                node_indices=member_list,
                end_timestamp_ns=end_ns,
                finished=bool(finished),
                _cached_points=levels[0][0],
                _cached_normals=levels[0][1],
                _cached_levels=levels,
                _hybrid_grid=grid,
                _gravity_count=gravity_count,
                _gravity_state=gravity,
            )
            frontend_submaps.append(frontend_submap)
            backend_submaps.append(frontend_submap.as_backend_submap())

        if stream.read(1):
            raise ValueError("native frontend state has trailing bytes")

    return NativeFrontendState(
        tuple(frontend_nodes),
        tuple(frontend_submaps),
        tuple(backend_nodes),
        tuple(backend_submaps),
    )
