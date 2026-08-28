from pathlib import Path
import sys
import tempfile

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_complete_slam import (  # noqa: E402
    load_frontend_state,
    topology_report,
)
from navvis_recon.surveyor_slam import (  # noqa: E402
    NodeId,
    Rigid3,
    Submap,
    TrajectoryNode,
)


def _pose(x: float = 0.0) -> Rigid3:
    return Rigid3(
        np.array([x, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )


def _metadata(
    *, finished: bool = True, gravity: np.ndarray | None = None
) -> tuple[tuple[TrajectoryNode, ...], tuple[Submap, ...]]:
    if gravity is None:
        gravity = np.array([0.0, 0.0, 9.80665])
    nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            index * 1_000_000_000,
            _pose(float(index)),
            _pose(float(index)),
            gravity.copy(),
        )
        for index in range(3)
    )
    submaps = (
        Submap(
            NodeId(0, 0),
            0,
            2_000_000_000,
            _pose(),
            (0, 1),
            finished,
            gravity.copy(),
        ),
    )
    return nodes, submaps


def _write_state(
    path: Path,
    *,
    include_strict_fields: bool,
    finished: bool = True,
    gravity: np.ndarray | None = None,
    hybrid_grid_cells: int = 10,
) -> None:
    if gravity is None:
        gravity = np.array([0.0, 0.0, 9.80665])
    arrays: dict[str, np.ndarray] = {
        "node_timestamps_ns": np.array(
            [0, 1_000_000_000, 2_000_000_000], dtype=np.int64
        ),
        "node_translations": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        "node_quaternions_xyzw": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0]), (3, 1)
        ),
        "submap_start_timestamps_ns": np.array([0], dtype=np.int64),
        "submap_end_timestamps_ns": np.array([2_000_000_000], dtype=np.int64),
        "submap_translations": np.zeros((1, 3)),
        "submap_quaternions_xyzw": np.array([[0.0, 0.0, 0.0, 1.0]]),
        "submap_membership_offsets": np.array([0, 2], dtype=np.int64),
        "submap_membership_indices": np.array([0, 1], dtype=np.int64),
        "loops": np.empty((0, 13), dtype=np.float64),
    }
    if include_strict_fields:
        arrays.update(
            {
                "state_schema_version": np.asarray(2, dtype=np.int64),
                "submap_finished": np.array([finished], dtype=np.bool_),
                "submap_gravity_observations": gravity.reshape(1, 3),
                "submap_hybrid_grid_cell_counts": np.array(
                    [hybrid_grid_cells], dtype=np.int64
                ),
            }
        )
    np.savez_compressed(path, **arrays)


def _check_npz_submap_state_is_generated_not_copied_from_frozen(tmp_path: Path):
    frozen_nodes, frozen_submaps = _metadata(
        finished=True, gravity=np.array([0.0, 0.0, 9.80665])
    )
    generated_gravity = np.array([1.0, 2.0, 3.0])
    state_path = tmp_path / "state.npz"
    _write_state(
        state_path,
        include_strict_fields=True,
        finished=False,
        gravity=generated_gravity,
        hybrid_grid_cells=11,
    )

    state = load_frontend_state(state_path, frozen_nodes, frozen_submaps)

    assert state.submaps[0].finished is False
    np.testing.assert_array_equal(
        state.submaps[0].gravity_observation, generated_gravity
    )
    report = topology_report(state, frozen_nodes, frozen_submaps, (), {NodeId(0, 0): 10})
    assert report["details"][0]["finished_exact"] is False
    assert report["details"][0]["gravity_bit_exact"] is False
    assert report["hybrid_grid"]["details"][0]["difference_cells"] == 1
    assert report["strict_all_exact"] is False


def _check_legacy_npz_marks_missing_submap_fields_unavailable(tmp_path: Path):
    frozen_nodes, frozen_submaps = _metadata(finished=False)
    state_path = tmp_path / "legacy_state.npz"
    _write_state(state_path, include_strict_fields=False)

    state = load_frontend_state(state_path, frozen_nodes, frozen_submaps)
    report = topology_report(state, frozen_nodes, frozen_submaps, (), {NodeId(0, 0): 10})

    assert report["status"] == "unavailable"
    assert report["strict_all_exact"] is False
    assert report["details"][0]["finished_exact"] is None
    assert report["details"][0]["gravity_bit_exact"] is None
    assert report["hybrid_grid"]["all_exact"] is None
    assert report["unavailable_generated_fields"] == [
        "submap_finished",
        "submap_gravity_observations",
        "submap_hybrid_grid_cell_counts",
    ]


def _check_strict_topology_reports_grid_and_constraint_graph_exact(tmp_path: Path):
    frozen_nodes, frozen_submaps = _metadata()
    state_path = tmp_path / "exact_state.npz"
    _write_state(state_path, include_strict_fields=True)
    state = load_frontend_state(state_path, frozen_nodes, frozen_submaps)

    report = topology_report(state, frozen_nodes, frozen_submaps, (), {NodeId(0, 0): 10})

    assert report["strict_all_exact"] is True
    assert report["hybrid_grid"]["all_exact"] is True
    graph = report["constraint_graph"]
    assert graph["generated"]["connected_components"] == 2
    assert graph["generated"]["component_sizes"] == [3, 1]
    assert graph["generated"]["isolated_vertices"] == 1
    assert graph["generated"]["isolated_vertex_ids"] == ["node:0:2"]
    assert graph["comparison"]["all_exact"] is True


def _check_invalid_hybrid_grid_counts_are_rejected(tmp_path: Path):
    frozen_nodes, frozen_submaps = _metadata()
    state_path = tmp_path / "bad_state.npz"
    _write_state(state_path, include_strict_fields=True)
    with np.load(state_path, allow_pickle=False) as state:
        arrays = {name: np.asarray(state[name]) for name in state.files}
    arrays["submap_hybrid_grid_cell_counts"] = np.array([-1], dtype=np.int64)
    np.savez_compressed(state_path, **arrays)

    try:
        load_frontend_state(state_path, frozen_nodes, frozen_submaps)
    except ValueError as error:
        assert "cannot be negative" in str(error)
    else:
        raise AssertionError("negative HybridGrid cell count was accepted")


def _run_in_temporary_directory(check) -> None:
    with tempfile.TemporaryDirectory(prefix="complete_slam_test_") as directory:
        check(Path(directory))


def test_npz_submap_state_is_generated_not_copied_from_frozen():
    _run_in_temporary_directory(
        _check_npz_submap_state_is_generated_not_copied_from_frozen
    )


def test_legacy_npz_marks_missing_submap_fields_unavailable():
    _run_in_temporary_directory(
        _check_legacy_npz_marks_missing_submap_fields_unavailable
    )


def test_strict_topology_reports_grid_and_constraint_graph_exact():
    _run_in_temporary_directory(
        _check_strict_topology_reports_grid_and_constraint_graph_exact
    )


def test_invalid_hybrid_grid_counts_are_rejected():
    _run_in_temporary_directory(_check_invalid_hybrid_grid_counts_are_rejected)


if __name__ == "__main__":
    tests = [
        value
        for name, value in globals().copy().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
