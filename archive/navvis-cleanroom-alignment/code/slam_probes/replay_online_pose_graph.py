#!/usr/bin/env python3
"""Replay the online SparsePoseGraph optimization cadence.

This diagnostic stops before the offline pose-graph optimization.  It
reconstructs the online graph state used to seed same-trajectory loop matching:
the installed G11 build runs Ceres every ``optimize_every_n_scans`` retained
nodes and once more when the trajectory finishes.  Old parameter blocks warm
start from the previous solve, and new nodes/submaps are initialized through
the currently matching submap's local-to-global transform.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_complete_slam import load_frontend_state  # noqa: E402
from navvis_recon.surveyor_slam import (  # noqa: E402
    FastImuPoseGraphResult,
    ImuCalibration,
    NodeId,
    Rigid3,
    Submap,
    TrajectoryNode,
    _message,
    _read_zip_messages,
    _rigid3,
    build_fast_imu_pose_graph,
    build_pose_graph,
    load_optimization_imu,
    load_submaps,
    load_trajectory_nodes,
    optimize_fast_imu_pose_graph_ceres,
)


def _angle_degrees(left: Rigid3, right: Rigid3) -> float:
    return float(np.degrees((left.rotation.inv() * right.rotation).magnitude()))


def _prefix_submaps(
    submaps: Sequence[Submap], nodes: Sequence[TrajectoryNode]
) -> tuple[Submap, ...]:
    last_timestamp_ns = nodes[-1].timestamp_ns
    node_count = len(nodes)
    output = []
    for submap in submaps:
        if submap.start_timestamp_ns > last_timestamp_ns:
            continue
        memberships = tuple(index for index in submap.node_indices if index < node_count)
        output.append(
            Submap(
                submap.submap_id,
                submap.start_timestamp_ns,
                min(submap.end_timestamp_ns, last_timestamp_ns),
                submap.local_pose,
                memberships,
                submap.finished and submap.node_indices[-1] < node_count,
                submap.gravity_observation,
            )
        )
    return tuple(output)


def _sample_window(samples, first_timestamp_ns: int, last_timestamp_ns: int):
    timestamps = tuple(sample.timestamp_ns for sample in samples)
    first = max(0, bisect_right(timestamps, first_timestamp_ns) - 1)
    last = min(len(samples), bisect_left(timestamps, last_timestamp_ns) + 1)
    return tuple(samples[first:last])


def _matching_submap(
    node_index: int, submaps: Sequence[Submap]
) -> Submap:
    memberships = [submap for submap in submaps if node_index in submap.node_indices]
    if not memberships:
        # A newly-started submap is created on the threshold node but receives
        # its first insertion on the following node.  The older map is still
        # the matching map at that boundary.
        older = [
            submap
            for submap in submaps
            if submap.start_timestamp_ns <= submaps[-1].start_timestamp_ns
            and submap.node_indices
            and submap.node_indices[0] <= node_index
        ]
        if not older:
            raise ValueError(f"node {node_index} has no matching submap")
        return older[0]
    return min(memberships, key=lambda submap: submap.submap_id.index)


def _extend_initial_state(
    nodes: Sequence[TrajectoryNode],
    submaps: Sequence[Submap],
    previous_count: int,
    previous_submaps: Sequence[Submap],
    previous: FastImuPoseGraphResult | None,
) -> tuple[tuple[Rigid3, ...], ImuCalibration]:
    if previous is None:
        poses = tuple(node.local_pose for node in nodes) + tuple(
            submap.local_pose for submap in submaps
        )
        return poses, ImuCalibration(gravity_magnitude=9.807232)

    previous_submap_pose = {
        submap.submap_id: previous.poses[previous_count + index]
        for index, submap in enumerate(previous_submaps)
    }
    poses: list[Rigid3] = list(previous.poses[:previous_count])
    for node in nodes[previous_count:]:
        matching = _matching_submap(node.node_id.index, submaps)
        matching_global = previous_submap_pose.get(matching.submap_id)
        if matching_global is None:
            # The only possible new map is initialized from the older active
            # map.  Use the oldest map containing this node as the bridge.
            bridge = min(
                (
                    candidate
                    for candidate in submaps
                    if candidate.submap_id in previous_submap_pose
                    and node.node_id.index in candidate.node_indices
                ),
                key=lambda candidate: candidate.submap_id.index,
            )
            matching_global = previous_submap_pose[bridge.submap_id].compose(
                bridge.local_pose.inverse()
            ).compose(matching.local_pose)
        poses.append(
            matching_global.compose(matching.local_pose.inverse()).compose(
                node.local_pose
            )
        )

    for submap in submaps:
        known = previous_submap_pose.get(submap.submap_id)
        if known is not None:
            poses.append(known)
            continue
        bridge = max(
            previous_submaps,
            key=lambda candidate: candidate.submap_id.index,
        )
        poses.append(
            previous_submap_pose[bridge.submap_id]
            .compose(bridge.local_pose.inverse())
            .compose(submap.local_pose)
        )
    assert previous.calibration is not None
    return tuple(poses), previous.calibration


def _historical_expected_pose(path: Path) -> Rigid3:
    messages = tuple(_read_zip_messages(path))
    if len(messages) != 1:
        raise ValueError("expected one historical loop constraint")
    payload = _message(messages[0][1], 1)
    return _rigid3(_message(payload, 7))


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    frozen_nodes = load_trajectory_nodes(args.nodes)
    frozen_submaps = load_submaps(args.submaps)
    state = load_frontend_state(args.state, frozen_nodes, frozen_submaps)
    samples = load_optimization_imu(args.optimization_data)
    expected = _historical_expected_pose(args.loops)

    solve_counts = list(
        range(args.optimize_every, len(state.nodes) + 1, args.optimize_every)
    )
    if not solve_counts or solve_counts[-1] != len(state.nodes):
        solve_counts.append(len(state.nodes))
    previous: FastImuPoseGraphResult | None = None
    previous_count = 0
    previous_submaps: tuple[Submap, ...] = ()
    solves = []
    for count in solve_counts:
        nodes = state.nodes[:count]
        submaps = _prefix_submaps(state.submaps, nodes)
        pose_graph = build_pose_graph(nodes, submaps, ())
        initial_poses, calibration = _extend_initial_state(
            nodes, submaps, previous_count, previous_submaps, previous
        )
        pose_graph = replace(pose_graph, initial_poses=initial_poses)
        sample_window = _sample_window(
            samples, nodes[0].timestamp_ns, nodes[-1].timestamp_ns
        )
        problem = build_fast_imu_pose_graph(
            pose_graph, nodes, sample_window, calibration
        )
        solve_work = args.work_dir / f"solve_{count:05d}"
        previous = optimize_fast_imu_pose_graph_ceres(
            problem,
            args.ceres_solver,
            solve_work,
            max_iterations=args.max_iterations,
            num_threads=args.solver_threads,
        )
        previous_count = count
        previous_submaps = submaps
        solves.append(
            {
                "node_count": count,
                "submap_count": len(submaps),
                "edge_count": len(problem.pose_graph.edges),
                "initial_cost": previous.initial_cost,
                "final_cost": previous.final_cost,
                "iterations": previous.iterations,
            }
        )

    if previous is None:
        raise ValueError("seed node occurs before the first online optimization")
    seed_node = state.nodes[args.seed_node]
    seed_submaps = _prefix_submaps(state.submaps, state.nodes)
    matching = _matching_submap(args.seed_node, seed_submaps)
    target = next(
        submap for submap in seed_submaps if submap.submap_id.index == args.target_submap
    )
    previous_submap_pose = {
        submap.submap_id: previous.poses[previous_count + index]
        for index, submap in enumerate(previous_submaps)
    }
    if args.seed_node < previous_count:
        node_global = previous.poses[args.seed_node]
    else:
        matching_global = previous_submap_pose[matching.submap_id]
        node_global = matching_global.compose(matching.local_pose.inverse()).compose(
            seed_node.local_pose
        )
    actual = previous_submap_pose[target.submap_id].between(node_global)
    translation_error = float(np.linalg.norm(actual.translation - expected.translation))
    rotation_error = _angle_degrees(actual, expected)

    if args.state_output is not None:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_output,
            node_count=np.array(previous_count, dtype=np.int64),
            node_translations=np.vstack(
                [pose.translation for pose in previous.poses[:previous_count]]
            ),
            node_quaternions_xyzw=np.vstack(
                [pose.quaternion_xyzw for pose in previous.poses[:previous_count]]
            ),
            submap_ids=np.array(
                [submap.submap_id.index for submap in previous_submaps], dtype=np.int64
            ),
            submap_translations=np.vstack(
                [
                    previous_submap_pose[submap.submap_id].translation
                    for submap in previous_submaps
                ]
            ),
            submap_quaternions_xyzw=np.vstack(
                [
                    previous_submap_pose[submap.submap_id].quaternion_xyzw
                    for submap in previous_submaps
                ]
            ),
        )

    return {
        "online_schedule": {
            "optimize_every_n_scans": args.optimize_every,
            "trigger_condition": (
                "retained_node_count reaches the configured period, plus "
                "FinishTrajectory"
            ),
            "solve_counts": solve_counts,
            "max_iterations": args.max_iterations,
            "solver_threads": args.solver_threads,
        },
        "solves": solves,
        "seed": {
            "target_submap": args.target_submap,
            "node": args.seed_node,
            "matching_submap": matching.submap_id.index,
            "translation": actual.translation.tolist(),
            "quaternion_xyzw": actual.quaternion_xyzw.tolist(),
            "historical_translation": expected.translation.tolist(),
            "historical_quaternion_xyzw": expected.quaternion_xyzw.tolist(),
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--submaps", type=Path, required=True)
    parser.add_argument("--loops", type=Path, required=True)
    parser.add_argument("--optimization-data", type=Path, required=True)
    parser.add_argument("--ceres-solver", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--optimize-every", type=int, default=321)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--solver-threads", type=int, default=7)
    parser.add_argument("--target-submap", type=int, default=0)
    parser.add_argument("--seed-node", type=int, default=2630)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
