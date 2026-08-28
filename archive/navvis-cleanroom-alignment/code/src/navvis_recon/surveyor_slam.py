"""SurveyorSLAM intermediate formats and sparse SE(3) pose-graph backend.

The installed NavVis SLAM programs retain sharded protobuf messages for every
trajectory node, submap and accepted loop closure.  The package does not ship
the corresponding ``.proto`` sources, so this module implements the small
protobuf wire subset needed by those messages.  Field assignments below were
validated against the 2026 G11 regression output and against the counts in the
original ``compute_trajectories`` log:

* 1,617 trajectory nodes;
* 6 retained submaps;
* 2,581 node-to-submap odometry constraints; and
* 15 accepted loop closures.

The optimizer is independent of the reader.  A future raw-lidar frontend can
emit the same :class:`PoseGraphEdge` objects without depending on reference
artifacts.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import ctypes
from dataclasses import dataclass, field, replace
import gzip
import os
from pathlib import Path
import struct
import subprocess
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def _native_solver_environment(solver: Path) -> dict[str, str]:
    """Resolve the bundled Ceres libraries, including indirect dependencies."""

    environment = os.environ.copy()
    search_root = solver.parent.parent
    library_directories: list[Path] = []
    for pattern in (
        "ceres*/root/usr/lib/libceres.so*",
        "ceres*/root/usr/lib/*/libspqr.so*",
    ):
        for library in sorted(search_root.glob(pattern)):
            directory = library.parent.resolve()
            if directory not in library_directories:
                library_directories.append(directory)
    if library_directories:
        inherited = environment.get("LD_LIBRARY_PATH")
        entries = [str(directory) for directory in library_directories]
        if inherited:
            entries.append(inherited)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
    return environment


@dataclass(frozen=True, slots=True)
class Rigid3:
    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if translation.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Rigid3 expects translation (3,) and quaternion (4,)")
        norm = np.linalg.norm(quaternion)
        if norm < 1.0e-12:
            raise ValueError("Rigid3 quaternion must be non-zero")
        object.__setattr__(self, "translation", translation)
        # Preserve serialized/internal coefficients.  SurveyorSLAM's first
        # raw-IMU pose intentionally carries the reciprocal norm of the
        # firmware quaternion until the first gravity correction.  SciPy
        # normalizes on conversion in ``rotation``; coefficient-sensitive
        # Eigen transforms use the stored values directly where required.
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_quat(self.quaternion_xyzw)

    def inverse(self) -> "Rigid3":
        inverse_rotation = self.rotation.inv()
        return Rigid3(
            inverse_rotation.apply(-self.translation), inverse_rotation.as_quat()
        )

    def compose(self, other: "Rigid3") -> "Rigid3":
        rotation = self.rotation
        return Rigid3(
            self.translation + rotation.apply(other.translation),
            (rotation * other.rotation).as_quat(),
        )

    def between(self, other: "Rigid3") -> "Rigid3":
        return self.inverse().compose(other)


def _raw_quaternion_product_xyzw(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Eigen-order Hamilton product without coefficient normalization."""

    lx, ly, lz, lw = (np.float64(value) for value in left)
    rx, ry, rz, rw = (np.float64(value) for value in right)
    return np.array(
        [
            (ly * rz + lw * rx) - (lz * ry - lx * rw),
            (ly * rw + lw * ry) + (lz * rx - lx * rz),
            (lw * rz - ly * rx) + (lx * ry + lz * rw),
            (lw * rw - ly * ry) - (lx * rx + lz * rz),
        ],
        dtype=np.float64,
    )


def _raw_quaternion_transform_vector(
    quaternion_xyzw: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Match Eigen ``Quaternion::_transformVector`` on raw coefficients."""

    qx, qy, qz, qw = (np.float64(value) for value in quaternion_xyzw)
    vx, vy, vz = (np.float64(value) for value in vector)
    tx = qy * vz - qz * vy
    tx += tx
    ty = qz * vx - qx * vz
    ty += ty
    tz = qx * vy - qy * vx
    tz += tz
    return np.array(
        [
            (qy * tz - qz * ty) + (qw * tx + vx),
            (qz * tx - qx * tz) + (qw * ty + vy),
            (qx * ty - qy * tx) + (vz + qw * tz),
        ],
        dtype=np.float64,
    )


def _raw_relative_pose(source: Rigid3, target: Rigid3) -> Rigid3:
    """Reproduce the installed ``source.inverse() * target`` coefficient path.

    The frontend stores near-unit quaternions without canonicalizing their
    sign or norm.  The graph residual functor receives those exact relative
    coefficients, so routing this operation through SciPy changes the Ceres
    problem even though it represents the same geometric rotation.
    """

    x, y, z, w = (np.float64(value) for value in source.quaternion_xyzw)
    squared_norm = (z * z + x * x) + (w * w + y * y)
    if squared_norm <= 0.0:
        raise ValueError("relative-pose source quaternion must be non-zero")
    inverse = np.array(
        [-x / squared_norm, -y / squared_norm, -z / squared_norm, w / squared_norm],
        dtype=np.float64,
    )
    return Rigid3(
        # Rigid3 evaluates ``source.inverse() * target`` as two independent
        # quaternion-vector products followed by addition.  Collapsing this
        # to one transform of the translation difference changes low bits.
        _raw_quaternion_transform_vector(inverse, -source.translation)
        + _raw_quaternion_transform_vector(inverse, target.translation),
        _raw_quaternion_product_xyzw(inverse, target.quaternion_xyzw),
    )


def _raw_inverse_pose(pose: Rigid3) -> Rigid3:
    """Invert one stored Eigen pose without normalizing its quaternion."""

    x, y, z, w = (np.float64(value) for value in pose.quaternion_xyzw)
    # Keep the same reduction order as the recovered Eigen binary.  This is
    # intentionally not ``np.dot`` because its SIMD reduction changes ulps.
    squared_norm = (z * z + x * x) + (w * w + y * y)
    if squared_norm <= 0.0:
        raise ValueError("pose quaternion must be non-zero")
    inverse = np.array(
        [-x / squared_norm, -y / squared_norm, -z / squared_norm, w / squared_norm],
        dtype=np.float64,
    )
    return Rigid3(
        _raw_quaternion_transform_vector(inverse, -pose.translation), inverse
    )


def _raw_compose_pose(left: Rigid3, right: Rigid3) -> Rigid3:
    """Compose stored Eigen poses while preserving their raw coefficients."""

    return Rigid3(
        left.translation
        + _raw_quaternion_transform_vector(
            left.quaternion_xyzw, right.translation
        ),
        _raw_quaternion_product_xyzw(
            left.quaternion_xyzw, right.quaternion_xyzw
        ),
    )


@dataclass(frozen=True, slots=True)
class NodeId:
    trajectory: int
    index: int


@dataclass(frozen=True, slots=True)
class TrajectoryNode:
    node_id: NodeId
    timestamp_ns: int
    local_pose: Rigid3
    global_pose: Rigid3
    gravity_observation: np.ndarray


@dataclass(frozen=True, slots=True)
class Submap:
    submap_id: NodeId
    start_timestamp_ns: int
    end_timestamp_ns: int
    local_pose: Rigid3
    node_indices: tuple[int, ...]
    finished: bool
    gravity_observation: np.ndarray


@dataclass(frozen=True, slots=True)
class LoopConstraint:
    submap_id: NodeId
    node_id: NodeId
    submap_from_node: Rigid3
    translation_weight: float
    rotation_weight: float
    valid: bool
    tag: int


@dataclass(frozen=True, slots=True)
class ImuIntrinsics:
    bias: np.ndarray
    scaling: np.ndarray
    cross_axis: np.ndarray


@dataclass(frozen=True, slots=True)
class OptimizationState:
    timestamps_ns: np.ndarray
    poses: tuple[Rigid3, ...]
    velocities: np.ndarray
    submap_poses: tuple[Rigid3, ...]
    gravity_magnitude: float
    imu_from_tracking: Rigid3
    linear_acceleration_intrinsics: ImuIntrinsics
    angular_velocity_intrinsics: ImuIntrinsics


@dataclass(frozen=True, slots=True)
class PoseGraphEdge:
    source: int
    target: int
    source_from_target: Rigid3
    translation_weight: float
    rotation_weight: float
    kind: str = "odometry"


@dataclass(frozen=True, slots=True)
class PoseGraphProblem:
    initial_poses: tuple[Rigid3, ...]
    edges: tuple[PoseGraphEdge, ...]
    fixed_vertices: tuple[int, ...]
    node_vertex: Mapping[NodeId, int]
    submap_vertex: Mapping[NodeId, int]


@dataclass(frozen=True, slots=True)
class PoseGraphResult:
    poses: tuple[Rigid3, ...]
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class ImuSample:
    timestamp_ns: int
    linear_acceleration: np.ndarray
    angular_velocity: np.ndarray
    delta_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_rotation_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )
    orientation_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )

    def __post_init__(self) -> None:
        vectors = {
            "linear_acceleration": (self.linear_acceleration, (3,)),
            "angular_velocity": (self.angular_velocity, (3,)),
            "delta_velocity": (self.delta_velocity, (3,)),
            "delta_rotation_xyzw": (self.delta_rotation_xyzw, (4,)),
            "orientation_xyzw": (self.orientation_xyzw, (4,)),
        }
        for name, (value, shape) in vectors.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite {shape} array")
            if shape == (4,):
                norm = np.linalg.norm(array)
                if norm < 1.0e-12:
                    raise ValueError(f"{name} quaternion must be non-zero")
                # The G11 firmware orientation is not exactly unit length and
                # the raw-IMU tracker deliberately consumes those coefficients
                # unchanged during initialization.  Delta rotations, however,
                # are mathematical rotations and remain normalized here.
                if name == "delta_rotation_xyzw":
                    array = array / norm
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class ImuCalibration:
    gravity_magnitude: float = 9.80665
    imu_from_tracking: Rigid3 = Rigid3(
        np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    )
    linear_acceleration_bias: np.ndarray | None = None
    linear_acceleration_scaling: np.ndarray | None = None
    linear_acceleration_cross_axis: np.ndarray | None = None
    angular_velocity_bias: np.ndarray | None = None
    angular_velocity_scaling: np.ndarray | None = None
    angular_velocity_cross_axis: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.gravity_magnitude) or self.gravity_magnitude <= 0:
            raise ValueError("gravity magnitude must be finite and positive")
        defaults = {
            "linear_acceleration_bias": np.zeros(3),
            "linear_acceleration_scaling": np.ones(3),
            "linear_acceleration_cross_axis": np.zeros(6),
            "angular_velocity_bias": np.zeros(3),
            "angular_velocity_scaling": np.ones(3),
            "angular_velocity_cross_axis": np.zeros(6),
        }
        for name, default in defaults.items():
            value = getattr(self, name)
            array = default if value is None else np.asarray(value, dtype=np.float64)
            if array.shape != default.shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite {default.shape} array")
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class ImuCalibrationOptions:
    """Installed ``optimization_problem_imu_intrinsics.lua`` priors.

    These are residual multipliers, not square-rooted covariance values.  The
    offline configuration probe records the decrypted reference chunks under
    ``work/slam_alignment_20260827/imu_intrinsics_decrypted_lua.log``.
    """

    gravity_magnitude: float = 9.807232
    gravity_prior_weight: float = 1.0e4
    imu_orientation_prior_weight: float = 5.0e4
    linear_acceleration_bias_prior_weight: tuple[float, float, float] = (
        1.0e5,
        1.0e5,
        1.0e2,
    )
    linear_acceleration_scaling_prior_weight: tuple[float, float, float] = (
        1.0e5,
        1.0e5,
        1.0e5,
    )
    angular_velocity_bias_prior_weight: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    angular_velocity_scaling_prior_weight: tuple[float, float, float] = (
        1.0e4,
        1.0e4,
        1.0e4,
    )


@dataclass(frozen=True, slots=True)
class ImuPreintegration:
    source_node: NodeId
    target_node: NodeId
    delta_t: float
    delta_rotation_xyzw: np.ndarray
    delta_velocity: np.ndarray
    delta_position: np.ndarray
    sample_count: int
    source_timestamp_ns: int
    target_timestamp_ns: int


@dataclass(frozen=True, slots=True)
class ImuPoseGraphProblem:
    pose_graph: PoseGraphProblem
    node_vertices: tuple[int, ...]
    initial_velocities: np.ndarray
    preintegrations: tuple[ImuPreintegration, ...]
    gravity_magnitude: float
    samples: tuple[ImuSample, ...] = ()
    calibration: ImuCalibration = field(default_factory=ImuCalibration)


@dataclass(frozen=True, slots=True)
class ImuPoseGraphResult:
    poses: tuple[Rigid3, ...]
    velocities: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    message: str
    calibration: ImuCalibration | None = None


@dataclass(frozen=True, slots=True)
class FastImuAccelerationFactor:
    first_node: NodeId
    second_node: NodeId
    third_node: NodeId
    delta_velocity: np.ndarray
    first_duration: float
    second_duration: float
    loss_duration: float


@dataclass(frozen=True, slots=True)
class FastImuRotationFactor:
    first_node: NodeId
    second_node: NodeId
    delta_rotation_xyzw: np.ndarray
    duration: float


@dataclass(frozen=True, slots=True)
class FastImuPoseGraphProblem:
    pose_graph: PoseGraphProblem
    node_vertices: tuple[int, ...]
    acceleration_factors: tuple[FastImuAccelerationFactor, ...]
    rotation_factors: tuple[FastImuRotationFactor, ...]
    calibration: ImuCalibration


@dataclass(frozen=True, slots=True)
class FastImuPoseGraphResult:
    poses: tuple[Rigid3, ...]
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    message: str
    calibration: ImuCalibration


@dataclass(frozen=True, slots=True)
class OnlineFastPoseGraphSnapshot:
    """One installed-cadence online Fast-IMU pose-graph solution."""

    node_count: int
    submaps: tuple[Submap, ...]
    result: FastImuPoseGraphResult


@dataclass(frozen=True, slots=True)
class _WireField:
    number: int
    wire_type: int
    value: int | float | bytes


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift <= 63:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def _wire_fields(data: bytes) -> tuple[_WireField, ...]:
    fields: list[_WireField] = []
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if number == 0:
            raise ValueError("protobuf field zero is invalid")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            if offset + 8 > len(data):
                raise ValueError("truncated protobuf fixed64")
            value = struct.unpack_from("<d", data, offset)[0]
            offset += 8
        elif wire_type == 2:
            size, offset = _read_varint(data, offset)
            if offset + size > len(data):
                raise ValueError("truncated protobuf bytes")
            value = data[offset : offset + size]
            offset += size
        elif wire_type == 5:
            if offset + 4 > len(data):
                raise ValueError("truncated protobuf fixed32")
            value = struct.unpack_from("<f", data, offset)[0]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        fields.append(_WireField(number, wire_type, value))
    return tuple(fields)


def _values(data: bytes, number: int) -> list[int | float | bytes]:
    return [field.value for field in _wire_fields(data) if field.number == number]


def _one(data: bytes, number: int) -> int | float | bytes:
    values = _values(data, number)
    if len(values) != 1:
        raise ValueError(f"expected one field {number}, got {len(values)}")
    return values[0]


def _message(data: bytes, number: int) -> bytes:
    value = _one(data, number)
    if not isinstance(value, bytes):
        raise ValueError(f"field {number} is not a protobuf message")
    return value


def _vector3(data: bytes) -> np.ndarray:
    return np.array([float(_one(data, field)) for field in (1, 2, 3)])


def _quaternion_xyzw(data: bytes) -> np.ndarray:
    return np.array([float(_one(data, field)) for field in (1, 2, 3, 4)])


def _rigid3(data: bytes) -> Rigid3:
    translation = _vector3(_message(data, 1))
    quaternion = _quaternion_xyzw(_message(data, 2))
    return Rigid3(translation, quaternion)


def _node_id(data: bytes) -> NodeId:
    return NodeId(int(_one(data, 1)), int(_one(data, 2)))


def parse_trajectory_node(data: bytes, node_id: NodeId) -> TrajectoryNode:
    # Repeated outer field 5 contains insertion/odometry-constraint IDs, not
    # the trajectory-node ID.  A node can have two entries while two submaps
    # overlap.  The stable node index is the sharded archive filename.
    node_data = _message(data, 1)
    return TrajectoryNode(
        node_id=node_id,
        timestamp_ns=int(_one(node_data, 1)),
        local_pose=_rigid3(_message(node_data, 8)),
        global_pose=_rigid3(_message(data, 2)),
        gravity_observation=_vector3(_message(node_data, 12)),
    )


def parse_submap(data: bytes) -> Submap:
    payload = _message(data, 1)
    indices = tuple(int(value) for value in _values(payload, 11))
    submap_index = int(_one(payload, 13))
    return Submap(
        submap_id=NodeId(0, submap_index),
        start_timestamp_ns=int(_one(payload, 8)),
        end_timestamp_ns=int(_one(payload, 9)),
        local_pose=_rigid3(_message(payload, 10)),
        node_indices=indices,
        finished=bool(_one(payload, 12)),
        gravity_observation=_vector3(_message(payload, 15)),
    )


def parse_loop_constraint(data: bytes) -> LoopConstraint:
    constraint_data = _message(_message(data, 1), 1)
    return LoopConstraint(
        submap_id=_node_id(_message(constraint_data, 1)),
        node_id=_node_id(_message(constraint_data, 2)),
        submap_from_node=_rigid3(_message(constraint_data, 3)),
        tag=int(_one(constraint_data, 5)),
        translation_weight=float(_one(constraint_data, 6)),
        rotation_weight=float(_one(constraint_data, 7)),
        valid=bool(_one(constraint_data, 8)),
    )


def _read_zip_messages(path: str | Path) -> Iterable[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".pb"):
                yield name, archive.read(name)


def load_trajectory_nodes(
    path: str | Path, *, limit: int | None = None
) -> tuple[TrajectoryNode, ...]:
    if limit is not None and limit < 1:
        raise ValueError("trajectory node limit must be positive")
    nodes: list[TrajectoryNode] = []
    for name, data in _read_zip_messages(path):
        try:
            index = int(Path(name).stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"cannot recover node index from {name}") from error
        nodes.append(parse_trajectory_node(data, NodeId(0, index)))
        if limit is not None and len(nodes) >= limit:
            break
    return tuple(nodes)


def load_submaps(path: str | Path) -> tuple[Submap, ...]:
    return tuple(parse_submap(data) for _, data in _read_zip_messages(path))


def load_loop_constraints(path: str | Path) -> tuple[LoopConstraint, ...]:
    return tuple(parse_loop_constraint(data) for _, data in _read_zip_messages(path))


def _imu_intrinsics(data: bytes) -> ImuIntrinsics:
    bias = _vector3(_message(data, 1))
    scaling = _vector3(_message(data, 2))
    cross_axis_message = _message(data, 3)
    cross_axis_values = np.array(
        [float(value) for value in _values(cross_axis_message, 3)], dtype=np.float64
    )
    if cross_axis_values.size != 6:
        raise ValueError("expected six IMU cross-axis parameters")
    return ImuIntrinsics(bias, scaling, cross_axis_values)


def load_optimization_state(path: str | Path) -> OptimizationState:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    fields = _wire_fields(raw)
    pose_states = [field.value for field in fields if field.number == 2]
    submap_states = [field.value for field in fields if field.number == 3]
    correction_states = [field.value for field in fields if field.number == 4]
    if not all(isinstance(value, bytes) for value in pose_states + submap_states):
        raise ValueError("optimization state contains non-message entries")
    if len(correction_states) != 1 or not isinstance(correction_states[0], bytes):
        raise ValueError("optimization state has no unique IMU correction")

    timestamps: list[int] = []
    poses: list[Rigid3] = []
    velocities: list[np.ndarray] = []
    for value in pose_states:
        assert isinstance(value, bytes)
        timestamps.append(int(_one(value, 1)))
        poses.append(_rigid3(_message(value, 2)))
        velocities.append(_vector3(_message(value, 3)))
    submap_poses = tuple(
        _rigid3(_message(value, 2))  # type: ignore[arg-type]
        for value in submap_states
    )
    correction = correction_states[0]
    assert isinstance(correction, bytes)
    return OptimizationState(
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        poses=tuple(poses),
        velocities=np.asarray(velocities, dtype=np.float64),
        submap_poses=submap_poses,
        imu_from_tracking=_rigid3(_message(correction, 1)),
        gravity_magnitude=float(_one(correction, 2)),
        linear_acceleration_intrinsics=_imu_intrinsics(_message(correction, 3)),
        angular_velocity_intrinsics=_imu_intrinsics(_message(correction, 4)),
    )


def imu_calibration_from_state(state: OptimizationState) -> ImuCalibration:
    """Convert the binary optimization-state correction into live IMU terms."""

    return ImuCalibration(
        gravity_magnitude=state.gravity_magnitude,
        imu_from_tracking=state.imu_from_tracking,
        linear_acceleration_bias=state.linear_acceleration_intrinsics.bias,
        linear_acceleration_scaling=state.linear_acceleration_intrinsics.scaling,
        linear_acceleration_cross_axis=state.linear_acceleration_intrinsics.cross_axis,
        angular_velocity_bias=state.angular_velocity_intrinsics.bias,
        angular_velocity_scaling=state.angular_velocity_intrinsics.scaling,
        angular_velocity_cross_axis=state.angular_velocity_intrinsics.cross_axis,
    )


def load_imu_rosbag(
    path: str | Path,
    *,
    topic: str = "/imu/imu_raw/data",
    end_timestamp_ns: int | None = None,
) -> tuple[ImuSample, ...]:
    """Read the 100 Hz raw IMU stream used by the vendor optimizer.

    ``rosbag`` is imported lazily because it is supplied by ROS rather than by
    this Python package.  Message header time is used, matching the timestamp
    visible in the serialized optimization inputs.
    """

    try:
        import rosbag  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("ROS1 rosbag Python bindings are required") from error
    samples: list[ImuSample] = []
    with rosbag.Bag(str(path)) as bag:
        for _, message, _ in bag.read_messages(topics=[topic]):
            stamp = message.header.stamp
            timestamp_ns = int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
            samples.append(
                ImuSample(
                    timestamp_ns,
                    np.array(
                        [
                            message.linear_acceleration.x,
                            message.linear_acceleration.y,
                            message.linear_acceleration.z,
                        ],
                        dtype=np.float64,
                    ),
                    np.array(
                        [
                            message.angular_velocity.x,
                            message.angular_velocity.y,
                            message.angular_velocity.z,
                        ],
                        dtype=np.float64,
                    ),
                    orientation_xyzw=np.array(
                        [
                            message.orientation.x,
                            message.orientation.y,
                            message.orientation.z,
                            message.orientation.w,
                        ],
                        dtype=np.float64,
                    ),
                )
            )
            # Retain the first sample beyond the requested endpoint so the
            # final lidar timestamp still has a complete interpolation
            # bracket during prefix regressions.
            if end_timestamp_ns is not None and timestamp_ns > end_timestamp_ns:
                break
    if len(samples) < 2 or any(
        second.timestamp_ns <= first.timestamp_ns
        for first, second in zip(samples, samples[1:])
    ):
        raise ValueError("IMU stream must contain strictly increasing samples")
    return tuple(samples)


def load_optimization_imu(path: str | Path) -> tuple[ImuSample, ...]:
    """Read the serialized IMU stream retained by the optimizer.

    Field 1 of ``TrajectoryOptimizationData`` is the repeated 100 Hz
    ``sensor.proto.ImuData`` stream.  Keeping its delta and orientation fields
    prevents a frozen optimization input from being silently replaced by a
    separately timestamped ROS bag stream.
    """

    raw = Path(path).read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    samples: list[ImuSample] = []
    for field_value in _values(raw, 1):
        if not isinstance(field_value, bytes):
            raise ValueError("optimization IMU entry is not a protobuf message")
        samples.append(
            ImuSample(
                int(_one(field_value, 1)),
                _vector3(_message(field_value, 2)),
                _vector3(_message(field_value, 3)),
                _vector3(_message(field_value, 4)),
                _quaternion_xyzw(_message(field_value, 5)),
                _quaternion_xyzw(_message(field_value, 6)),
            )
        )
    if len(samples) < 2 or any(
        second.timestamp_ns <= first.timestamp_ns
        for first, second in zip(samples, samples[1:])
    ):
        raise ValueError("optimization IMU stream must be strictly increasing")
    return tuple(samples)


def _correct_imu_sample(
    sample: ImuSample, calibration: ImuCalibration
) -> tuple[np.ndarray, np.ndarray]:
    # The installed Exact factor applies the optimized IMU orientation in the
    # stored direction (not its inverse).  The six cross-axis terms are zero
    # for the G11 reference, but retaining the triangular correction here
    # preserves the complete 15-parameter-block layout.
    def apply_intrinsics(
        value: np.ndarray, bias: np.ndarray, scaling: np.ndarray, cross: np.ndarray
    ) -> np.ndarray:
        corrected = (value - bias) * scaling
        matrix = np.array(
            [
                [1.0, cross[0], cross[1]],
                [cross[2], 1.0, cross[3]],
                [cross[4], cross[5], 1.0],
            ]
        )
        return matrix @ corrected

    imu_rotation = calibration.imu_from_tracking.rotation
    acceleration = imu_rotation.apply(
        apply_intrinsics(
            sample.linear_acceleration,
            calibration.linear_acceleration_bias,
            calibration.linear_acceleration_scaling,
            calibration.linear_acceleration_cross_axis,
        )
    )
    angular_velocity = imu_rotation.apply(
        apply_intrinsics(
            sample.angular_velocity,
            calibration.angular_velocity_bias,
            calibration.angular_velocity_scaling,
            calibration.angular_velocity_cross_axis,
        )
    )
    return acceleration, angular_velocity


def preintegrate_imu(
    samples: Sequence[ImuSample],
    source_timestamp_ns: int,
    target_timestamp_ns: int,
    calibration: ImuCalibration,
    *,
    source_node: NodeId = NodeId(0, 0),
    target_node: NodeId = NodeId(0, 1),
) -> ImuPreintegration:
    """Reproduce the installed Exact factor's endpoint-interpolated integral."""

    if target_timestamp_ns <= source_timestamp_ns:
        raise ValueError("IMU integration interval must be positive")
    if len(samples) < 2:
        raise ValueError("IMU integration needs at least two samples")
    sample_times = tuple(sample.timestamp_ns for sample in samples)
    if any(second <= first for first, second in zip(sample_times, sample_times[1:])):
        raise ValueError("IMU samples must be strictly increasing")
    if source_timestamp_ns < sample_times[0] or target_timestamp_ns > sample_times[-1]:
        raise ValueError("IMU samples do not bracket the integration interval")

    corrected = tuple(_correct_imu_sample(sample, calibration) for sample in samples)
    return _preintegrate_corrected_imu(
        sample_times,
        corrected,
        source_timestamp_ns,
        target_timestamp_ns,
        source_node=source_node,
        target_node=target_node,
    )


def _preintegrate_corrected_imu(
    sample_times: Sequence[int],
    corrected: Sequence[tuple[np.ndarray, np.ndarray]],
    source_timestamp_ns: int,
    target_timestamp_ns: int,
    *,
    source_node: NodeId,
    target_node: NodeId,
) -> ImuPreintegration:
    """Integrate one node interval from a shared corrected IMU cache."""

    first_inside = bisect_right(sample_times, source_timestamp_ns)
    last_inside = bisect_left(sample_times, target_timestamp_ns)
    times = [source_timestamp_ns]
    times.extend(sample_times[first_inside:last_inside])
    times.append(target_timestamp_ns)

    def value_at(timestamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
        index = bisect_left(sample_times, timestamp_ns)
        if index < len(sample_times) and sample_times[index] == timestamp_ns:
            return corrected[index]
        before = index - 1
        after = index
        alpha = (timestamp_ns - sample_times[before]) / (
            sample_times[after] - sample_times[before]
        )
        return tuple(
            (1.0 - alpha) * corrected[before][component]
            + alpha * corrected[after][component]
            for component in (0, 1)
        )  # type: ignore[return-value]

    values = [value_at(timestamp) for timestamp in times]

    delta_rotation = Rotation.identity()
    delta_velocity = np.zeros(3)
    delta_position = np.zeros(3)
    for index in range(len(times) - 1):
        dt = float(times[index + 1] - times[index]) / 1.0e9
        acceleration = 0.5 * (values[index][0] + values[index + 1][0])
        angular_velocity = 0.5 * (values[index][1] + values[index + 1][1])
        half_step_rotation = delta_rotation * Rotation.from_rotvec(
            angular_velocity * (0.5 * dt)
        )
        acceleration_source = half_step_rotation.apply(acceleration)
        delta_position += delta_velocity * dt + 0.5 * acceleration_source * dt * dt
        delta_velocity += acceleration_source * dt
        delta_rotation = delta_rotation * Rotation.from_rotvec(angular_velocity * dt)
    return ImuPreintegration(
        source_node,
        target_node,
        float(target_timestamp_ns - source_timestamp_ns) / 1.0e9,
        delta_rotation.as_quat(),
        delta_velocity,
        delta_position,
        max(0, last_inside - first_inside),
        source_timestamp_ns,
        target_timestamp_ns,
    )


class _FastImuNativeIntegrator:
    """Own contiguous Stage1 samples and call the exact C++ factor kernel."""

    def __init__(self, samples: Sequence[ImuSample]) -> None:
        from .surveyor_frontend import _load_slam_frontend_native

        self._timestamps = np.ascontiguousarray(
            [sample.timestamp_ns for sample in samples], dtype=np.int64
        )
        self._accelerations = np.ascontiguousarray(
            [sample.linear_acceleration for sample in samples], dtype=np.float64
        )
        self._angular_velocities = np.ascontiguousarray(
            [sample.angular_velocity for sample in samples], dtype=np.float64
        )
        self._library = _load_slam_frontend_native()
        self._timestamp_pointer = self._timestamps.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int64)
        )
        self._acceleration_pointer = self._accelerations.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        self._angular_velocity_pointer = self._angular_velocities.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @staticmethod
    def _double_pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def integrate(
        self, source_timestamp_ns: int, target_timestamp_ns: int
    ) -> tuple[np.ndarray, np.ndarray]:
        quaternion = np.empty(4, dtype=np.float64)
        velocity = np.empty(3, dtype=np.float64)
        status = self._library.navvis_recon_slam_fast_imu_integrate(
            len(self._timestamps),
            self._timestamp_pointer,
            self._acceleration_pointer,
            self._angular_velocity_pointer,
            source_timestamp_ns,
            target_timestamp_ns,
            self._double_pointer(quaternion),
            self._double_pointer(velocity),
        )
        if status:
            raise RuntimeError(
                f"native Fast IMU integration failed with status {status}"
            )
        return quaternion, velocity

    def acceleration_measurement(
        self,
        first_timestamp_ns: int,
        second_timestamp_ns: int,
        third_timestamp_ns: int,
    ) -> np.ndarray:
        measurement = np.empty(3, dtype=np.float64)
        status = (
            self._library.navvis_recon_slam_fast_imu_acceleration_measurement(
                len(self._timestamps),
                self._timestamp_pointer,
                self._acceleration_pointer,
                self._angular_velocity_pointer,
                first_timestamp_ns,
                second_timestamp_ns,
                third_timestamp_ns,
                self._double_pointer(measurement),
            )
        )
        if status:
            raise RuntimeError(
                "native Fast IMU acceleration measurement failed with "
                f"status {status}"
            )
        return measurement


def _integrate_fast_imu(
    sample_times: Sequence[int],
    corrected: Sequence[tuple[np.ndarray, np.ndarray]],
    source_timestamp_ns: int,
    target_timestamp_ns: int,
) -> tuple[Rotation, np.ndarray]:
    """Integrate the pose-only Stage1 IMU delta.

    The installed pre-intrinsics path walks the original IMU sample pairs.  It
    clips the linearly interpolated angular-velocity integral to the requested
    interval, updates orientation, rotates the two *raw sample* accelerations,
    and only then clips their trapezoid.  In particular, it does not first
    interpolate acceleration at a requested interval boundary.  That ordering
    matters at every scan midpoint and is observable at the sub-micrometre
    level in the complete Stage1 solve.

    This differs deliberately from Stage2's position/velocity preintegrator,
    whose acceleration is evaluated at a rotational half step.
    """

    if target_timestamp_ns <= source_timestamp_ns:
        raise ValueError("fast IMU integration interval must be positive")
    if source_timestamp_ns < sample_times[0] or target_timestamp_ns > sample_times[-1]:
        raise ValueError("IMU samples do not bracket the fast integration interval")

    def clipped_linear_integral(
        first: np.ndarray,
        second: np.ndarray,
        full_duration: float,
        left_clip: float,
        right_clip: float,
    ) -> np.ndarray:
        """Match the installed helper's scalar operation order."""

        left_ratio = left_clip / full_duration
        right_ratio = right_clip / full_duration
        first_boundary = left_ratio * second + (1.0 - left_ratio) * first
        second_boundary = right_ratio * first + (1.0 - right_ratio) * second
        active_duration = full_duration - left_clip - right_clip
        return (first_boundary + second_boundary) * 0.5 * active_duration

    delta_rotation = Rotation.identity()
    delta_velocity = np.zeros(3)
    index = bisect_right(sample_times, source_timestamp_ns) - 1
    while sample_times[index] < target_timestamp_ns:
        first_timestamp_ns = sample_times[index]
        second_timestamp_ns = sample_times[index + 1]
        full_duration = float(second_timestamp_ns - first_timestamp_ns) / 1.0e9
        left_clip = float(
            max(source_timestamp_ns, first_timestamp_ns) - first_timestamp_ns
        ) / 1.0e9
        right_clip = float(
            second_timestamp_ns - min(target_timestamp_ns, second_timestamp_ns)
        ) / 1.0e9
        first_acceleration, first_angular_velocity = corrected[index]
        second_acceleration, second_angular_velocity = corrected[index + 1]
        delta_angle = clipped_linear_integral(
            first_angular_velocity,
            second_angular_velocity,
            full_duration,
            left_clip,
            right_clip,
        )
        next_rotation = delta_rotation * Rotation.from_rotvec(delta_angle)
        first_rotated_acceleration = delta_rotation.apply(first_acceleration)
        second_rotated_acceleration = next_rotation.apply(second_acceleration)
        delta_velocity += clipped_linear_integral(
            first_rotated_acceleration,
            second_rotated_acceleration,
            full_duration,
            left_clip,
            right_clip,
        )
        delta_rotation = next_rotation
        index += 1
    return delta_rotation, delta_velocity


def build_fast_imu_pose_graph(
    pose_graph: PoseGraphProblem,
    nodes: Sequence[TrajectoryNode],
    samples: Sequence[ImuSample],
    calibration: ImuCalibration,
    initial_state: OptimizationState | None = None,
) -> FastImuPoseGraphProblem:
    """Build the installed pre-intrinsics (Stage1) IMU graph.

    Stage1 does not introduce velocity blocks.  Its acceleration factor spans
    three consecutive node translations and one center-node rotation.  The
    IMU delta velocity covers the two scan midpoints and is expressed in the
    IMU frame at the center node.  Adjacent node rotations receive a separate
    delta-rotation factor.
    """

    ordered_nodes = sorted(nodes, key=lambda node: node.timestamp_ns)
    if len(ordered_nodes) < 3:
        raise ValueError("fast IMU graph needs at least three trajectory nodes")
    node_vertices = tuple(
        pose_graph.node_vertex[node.node_id] for node in ordered_nodes
    )
    if initial_state is not None:
        timestamps = np.asarray(
            [node.timestamp_ns for node in ordered_nodes], dtype=np.int64
        )
        if not np.array_equal(initial_state.timestamps_ns, timestamps):
            raise ValueError("initial optimization state timestamps differ from nodes")
        submap_count = len(pose_graph.initial_poses) - len(ordered_nodes)
        if len(initial_state.submap_poses) != submap_count:
            raise ValueError("initial optimization state has a different submap count")
        pose_graph = PoseGraphProblem(
            tuple(initial_state.poses) + tuple(initial_state.submap_poses),
            pose_graph.edges,
            pose_graph.fixed_vertices,
            pose_graph.node_vertex,
            pose_graph.submap_vertex,
        )

    sample_times = tuple(sample.timestamp_ns for sample in samples)
    if len(sample_times) < 2 or any(
        second <= first for first, second in zip(sample_times, sample_times[1:])
    ):
        raise ValueError("IMU samples must be strictly increasing")
    native_integrator = _FastImuNativeIntegrator(samples)

    rotations: list[FastImuRotationFactor] = []
    for first, second in zip(ordered_nodes, ordered_nodes[1:]):
        delta_rotation_xyzw, _ = native_integrator.integrate(
            first.timestamp_ns, second.timestamp_ns
        )
        duration = float(second.timestamp_ns - first.timestamp_ns) / 1.0e9
        rotations.append(
            FastImuRotationFactor(
                first.node_id,
                second.node_id,
                delta_rotation_xyzw,
                duration,
            )
        )

    accelerations: list[FastImuAccelerationFactor] = []
    for index, (first, second, third) in enumerate(
        zip(ordered_nodes, ordered_nodes[1:], ordered_nodes[2:])
    ):
        first_duration_ns = second.timestamp_ns - first.timestamp_ns
        second_duration_ns = third.timestamp_ns - second.timestamp_ns
        first_center_ns = first.timestamp_ns + first_duration_ns // 2
        second_center_ns = second.timestamp_ns + second_duration_ns // 2
        measurement = native_integrator.acceleration_measurement(
            first.timestamp_ns, second.timestamp_ns, third.timestamp_ns
        )
        accelerations.append(
            FastImuAccelerationFactor(
                first.node_id,
                second.node_id,
                third.node_id,
                measurement,
                float(first_duration_ns) / 1.0e9,
                float(second_duration_ns) / 1.0e9,
                float(second_center_ns - first_center_ns) / 1.0e9,
            )
        )

    return FastImuPoseGraphProblem(
        pose_graph,
        node_vertices,
        tuple(accelerations),
        tuple(rotations),
        calibration,
    )


def build_imu_pose_graph(
    pose_graph: PoseGraphProblem,
    nodes: Sequence[TrajectoryNode],
    samples: Sequence[ImuSample],
    calibration: ImuCalibration,
    initial_state: OptimizationState | None = None,
    *,
    reuse_initial_state_velocities: bool = False,
) -> ImuPoseGraphProblem:
    ordered_nodes = sorted(nodes, key=lambda node: node.timestamp_ns)
    if len(ordered_nodes) < 2:
        raise ValueError("IMU graph needs at least two trajectory nodes")
    node_vertices = tuple(pose_graph.node_vertex[node.node_id] for node in ordered_nodes)
    if initial_state is not None:
        if len(initial_state.poses) != len(ordered_nodes):
            raise ValueError("initial optimization state has a different node count")
        submap_count = len(pose_graph.initial_poses) - len(ordered_nodes)
        if len(initial_state.submap_poses) != submap_count:
            raise ValueError("initial optimization state has a different submap count")
        pose_graph = PoseGraphProblem(
            tuple(initial_state.poses) + tuple(initial_state.submap_poses),
            pose_graph.edges,
            pose_graph.fixed_vertices,
            pose_graph.node_vertex,
            pose_graph.submap_vertex,
        )
    sample_times = tuple(sample.timestamp_ns for sample in samples)
    if len(sample_times) < 2 or any(
        second <= first for first, second in zip(sample_times, sample_times[1:])
    ):
        raise ValueError("IMU samples must be strictly increasing")
    corrected_samples = tuple(
        _correct_imu_sample(sample, calibration) for sample in samples
    )
    preintegrations = tuple(
        _preintegrate_corrected_imu(
            sample_times,
            corrected_samples,
            first.timestamp_ns,
            second.timestamp_ns,
            source_node=first.node_id,
            target_node=second.node_id,
        )
        for first, second in zip(ordered_nodes, ordered_nodes[1:])
    )
    if initial_state is not None:
        node_timestamps = np.asarray(
            [node.timestamp_ns for node in ordered_nodes], dtype=np.int64
        )
        if not np.array_equal(initial_state.timestamps_ns, node_timestamps):
            raise ValueError("initial optimization state timestamps differ from nodes")
    if initial_state is not None and reuse_initial_state_velocities:
        velocities = np.asarray(initial_state.velocities, dtype=np.float64).copy()
    else:
        timestamps_ns = np.asarray(
            [node.timestamp_ns for node in ordered_nodes], dtype=np.int64
        )
        positions = np.array(
            [pose_graph.initial_poses[vertex].translation for vertex in node_vertices]
        )
        # ToSeconds subtracts integer timestamps before converting the duration
        # to binary64. Converting epoch-scale nanoseconds first discards their
        # low bits and perturbs every Stage2 initial velocity.
        durations_seconds = np.diff(timestamps_ns).astype(np.float64) / 1.0e9
        segment_velocities = np.diff(positions, axis=0) / durations_seconds[:, None]
        # Binary capture shows backward differences, with the first segment copied
        # into both endpoint velocity blocks of the first factor.
        velocities = np.vstack((segment_velocities[0], segment_velocities))
    return ImuPoseGraphProblem(
        pose_graph,
        node_vertices,
        velocities,
        preintegrations,
        calibration.gravity_magnitude,
        tuple(samples),
        calibration,
    )


def build_pose_graph(
    nodes: Sequence[TrajectoryNode],
    submaps: Sequence[Submap],
    loops: Sequence[LoopConstraint],
    *,
    odometry_translation_weight: float = 500.0,
    odometry_rotation_weight: float = 1600.0,
) -> PoseGraphProblem:
    """Build the vendor graph topology from node/submap memberships.

    Each submap stores every node inserted into it.  Their membership count is
    the ``odometry constraints`` count printed by the original optimizer; it
    is intentionally not replaced by a chain of adjacent-node edges.
    """

    if not nodes or not submaps:
        raise ValueError("pose graph needs nodes and submaps")
    if odometry_translation_weight <= 0 or odometry_rotation_weight <= 0:
        raise ValueError("odometry weights must be positive")

    ordered_nodes = sorted(nodes, key=lambda node: (node.node_id.trajectory, node.node_id.index))
    ordered_submaps = sorted(
        submaps, key=lambda submap: (submap.submap_id.trajectory, submap.submap_id.index)
    )
    node_vertex = {node.node_id: index for index, node in enumerate(ordered_nodes)}
    submap_vertex = {
        submap.submap_id: len(ordered_nodes) + index
        for index, submap in enumerate(ordered_submaps)
    }
    initial = tuple(node.local_pose for node in ordered_nodes) + tuple(
        submap.local_pose for submap in ordered_submaps
    )
    node_by_id = {node.node_id: node for node in ordered_nodes}

    memberships_by_node: dict[NodeId, list[Submap]] = {}
    for submap in ordered_submaps:
        for node_index in submap.node_indices:
            node_id = NodeId(submap.submap_id.trajectory, node_index)
            if node_id not in node_by_id:
                raise ValueError(f"submap references missing node {node_id}")
            memberships_by_node.setdefault(node_id, []).append(submap)

    edges: list[PoseGraphEdge] = []
    # Constraints are enqueued when each retained node is inserted. During an
    # overlap this means old-submap then new-submap for the same node, not all
    # memberships of one submap followed by the next. The order changes Ceres'
    # parallel reduction residues and is observable at sub-micrometre scale.
    for node in ordered_nodes:
        for submap in memberships_by_node.get(node.node_id, ()):
            edges.append(
                PoseGraphEdge(
                    submap_vertex[submap.submap_id],
                    node_vertex[node.node_id],
                    _raw_relative_pose(submap.local_pose, node.local_pose),
                    odometry_translation_weight,
                    odometry_rotation_weight,
                    "odometry",
                )
            )
    for loop in loops:
        if not loop.valid:
            continue
        if loop.submap_id not in submap_vertex or loop.node_id not in node_vertex:
            raise ValueError("loop constraint references a missing graph vertex")
        edges.append(
            PoseGraphEdge(
                submap_vertex[loop.submap_id],
                node_vertex[loop.node_id],
                loop.submap_from_node,
                loop.translation_weight,
                loop.rotation_weight,
                "loop",
            )
        )
    first_submap = min(submap_vertex.values())
    return PoseGraphProblem(
        initial, tuple(edges), (first_submap,), node_vertex, submap_vertex
    )


def edge_residuals(
    poses: Sequence[Rigid3], edges: Sequence[PoseGraphEdge]
) -> np.ndarray:
    residuals = np.empty((len(edges), 6), dtype=np.float64)
    for index, edge in enumerate(edges):
        predicted = poses[edge.source].between(poses[edge.target])
        residuals[index, :3] = (
            predicted.translation - edge.source_from_target.translation
        ) * edge.translation_weight
        measured_rotation = edge.source_from_target.rotation
        residuals[index, 3:] = (
            measured_rotation.inv() * predicted.rotation
        ).as_rotvec() * edge.rotation_weight
    return residuals


def optimize_pose_graph(
    problem: PoseGraphProblem,
    *,
    max_iterations: int = 200,
    robust_loss: str = "huber",
    robust_scale: float = 1.0,
) -> PoseGraphResult:
    """Optimize a sparse SE(3) graph with fixed gauge vertices.

    Rotation increments are represented in the tangent space around each
    initial orientation.  This keeps the solve well-conditioned for the small
    loop corrections observed in the binary output and avoids quaternion norm
    constraints.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    fixed = set(problem.fixed_vertices)
    variable_vertices = [
        index for index in range(len(problem.initial_poses)) if index not in fixed
    ]
    variable_offset = {vertex: 6 * index for index, vertex in enumerate(variable_vertices)}
    variable_vertex_array = np.asarray(variable_vertices, dtype=np.int64)
    initial_translations = np.vstack(
        [pose.translation for pose in problem.initial_poses]
    )
    initial_quaternions = np.vstack(
        [pose.quaternion_xyzw for pose in problem.initial_poses]
    )
    variable_initial_rotations = Rotation.from_quat(
        initial_quaternions[variable_vertex_array]
    )
    x0 = np.empty(6 * len(variable_vertices), dtype=np.float64)
    for vertex, offset in variable_offset.items():
        x0[offset : offset + 3] = problem.initial_poses[vertex].translation
        x0[offset + 3 : offset + 6] = 0.0

    def unpack_arrays(
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Rotation]:
        translations = initial_translations.copy()
        quaternions = initial_quaternions.copy()
        pose_parameters = parameters.reshape((-1, 6))
        translations[variable_vertex_array] = pose_parameters[:, :3]
        quaternions[variable_vertex_array] = (
            Rotation.from_rotvec(pose_parameters[:, 3:])
            * variable_initial_rotations
        ).as_quat()
        return translations, quaternions, Rotation.from_quat(quaternions)

    edge_source = np.fromiter((edge.source for edge in problem.edges), dtype=np.int64)
    edge_target = np.fromiter((edge.target for edge in problem.edges), dtype=np.int64)
    edge_translation = np.vstack(
        [edge.source_from_target.translation for edge in problem.edges]
    )
    edge_rotation_inverse = Rotation.from_quat(
        np.vstack([edge.source_from_target.quaternion_xyzw for edge in problem.edges])
    ).inv()
    edge_translation_weight = np.fromiter(
        (edge.translation_weight for edge in problem.edges), dtype=np.float64
    )
    edge_rotation_weight = np.fromiter(
        (edge.rotation_weight for edge in problem.edges), dtype=np.float64
    )

    def fun(parameters: np.ndarray) -> np.ndarray:
        translations, _, rotations = unpack_arrays(parameters)
        source_rotation = rotations[edge_source]
        predicted_translation = source_rotation.inv().apply(
            translations[edge_target] - translations[edge_source]
        )
        predicted_rotation = source_rotation.inv() * rotations[edge_target]
        return np.column_stack(
            (
                (predicted_translation - edge_translation)
                * edge_translation_weight[:, None],
                (edge_rotation_inverse * predicted_rotation).as_rotvec()
                * edge_rotation_weight[:, None],
            )
        ).reshape(-1)

    sparsity = lil_matrix((6 * len(problem.edges), len(x0)), dtype=np.int8)
    for edge_index, edge in enumerate(problem.edges):
        rows = slice(6 * edge_index, 6 * edge_index + 6)
        for vertex in (edge.source, edge.target):
            if vertex in variable_offset:
                offset = variable_offset[vertex]
                sparsity[rows, offset : offset + 6] = 1

    initial_residual = fun(x0)
    result = least_squares(
        fun,
        x0,
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss=robust_loss,
        f_scale=robust_scale,
        max_nfev=max_iterations,
        x_scale="jac",
    )
    final_residual = fun(result.x)
    translations, quaternions, _ = unpack_arrays(result.x)
    return PoseGraphResult(
        poses=tuple(
            Rigid3(translation, quaternion)
            for translation, quaternion in zip(translations, quaternions)
        ),
        initial_cost=0.5 * float(initial_residual @ initial_residual),
        final_cost=0.5 * float(final_residual @ final_residual),
        iterations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def optimize_fast_imu_pose_graph(
    problem: FastImuPoseGraphProblem,
    *,
    max_iterations: int = 200,
    acceleration_weight: float = 50.0,
    rotation_weight: float = 10.0,
    calibration_options: ImuCalibrationOptions = ImuCalibrationOptions(),
) -> FastImuPoseGraphResult:
    """Optimize the installed Stage1 pose-only IMU backend.

    The factor weights are normalized by their support duration.  Binary
    residual capture gives ``acceleration_weight / (center_1 - center_0)`` for
    the three-pose acceleration factor and ``rotation_weight / dt`` for the
    adjacent rotation factor.  The center interval is retained separately
    because integer-nanosecond midpoint rounding can differ from the mean of
    the neighboring pose durations by 0.5 ns.  Gravity magnitude and IMU
    orientation are shared live parameter blocks with the same priors used by
    Stage2.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if acceleration_weight <= 0 or rotation_weight <= 0:
        raise ValueError("fast IMU weights must be positive")

    pose_problem = problem.pose_graph
    fixed = set(pose_problem.fixed_vertices)
    gauge_tilt_vertex = min(fixed)
    variable_vertices = [
        index for index in range(len(pose_problem.initial_poses)) if index not in fixed
    ]
    pose_offset = {vertex: 6 * index for index, vertex in enumerate(variable_vertices)}
    gauge_tilt_offset = 6 * len(variable_vertices)
    gravity_offset = gauge_tilt_offset + 2
    orientation_offset = gravity_offset + 1
    parameter_count = orientation_offset + 3

    initial_translations = np.vstack(
        [pose.translation for pose in pose_problem.initial_poses]
    )
    initial_quaternions = np.vstack(
        [pose.quaternion_xyzw for pose in pose_problem.initial_poses]
    )
    variable_vertex_array = np.asarray(variable_vertices, dtype=np.int64)
    variable_initial_rotations = Rotation.from_quat(
        initial_quaternions[variable_vertex_array]
    )
    gauge_initial_rotation = Rotation.from_quat(
        initial_quaternions[gauge_tilt_vertex]
    )
    initial_imu_rotation = problem.calibration.imu_from_tracking.rotation

    x0 = np.empty(parameter_count, dtype=np.float64)
    for vertex, offset in pose_offset.items():
        x0[offset : offset + 3] = pose_problem.initial_poses[vertex].translation
        x0[offset + 3 : offset + 6] = 0.0
    x0[gauge_tilt_offset : gauge_tilt_offset + 2] = 0.0
    x0[gravity_offset] = problem.calibration.gravity_magnitude
    x0[orientation_offset : orientation_offset + 3] = 0.0

    def unpack(
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, Rotation, float, Rotation]:
        translations = initial_translations.copy()
        quaternions = initial_quaternions.copy()
        pose_parameters = parameters[:gauge_tilt_offset].reshape((-1, 6))
        translations[variable_vertex_array] = pose_parameters[:, :3]
        quaternions[variable_vertex_array] = (
            Rotation.from_rotvec(pose_parameters[:, 3:])
            * variable_initial_rotations
        ).as_quat()
        gauge_tilt = Rotation.from_rotvec(
            [parameters[gauge_tilt_offset], parameters[gauge_tilt_offset + 1], 0.0]
        )
        # The installed constant-yaw local parameterization applies the
        # two-axis increment on the right.  The frozen Stage1 first submap has
        # a 1.9e-10 degree right-relative yaw change, while a left increment
        # produces a visible world-frame z component.
        quaternions[gauge_tilt_vertex] = (
            gauge_initial_rotation * gauge_tilt
        ).as_quat()
        imu_rotation = (
            Rotation.from_rotvec(parameters[orientation_offset : orientation_offset + 3])
            * initial_imu_rotation
        )
        return (
            translations,
            Rotation.from_quat(quaternions),
            float(parameters[gravity_offset]),
            imu_rotation,
        )

    graph_source = np.fromiter(
        (edge.source for edge in pose_problem.edges), dtype=np.int64
    )
    graph_target = np.fromiter(
        (edge.target for edge in pose_problem.edges), dtype=np.int64
    )
    graph_translation = np.vstack(
        [edge.source_from_target.translation for edge in pose_problem.edges]
    )
    graph_rotation_inverse = Rotation.from_quat(
        np.vstack(
            [edge.source_from_target.quaternion_xyzw for edge in pose_problem.edges]
        )
    ).inv()
    graph_translation_weight = np.fromiter(
        (edge.translation_weight for edge in pose_problem.edges), dtype=np.float64
    )
    graph_rotation_weight = np.fromiter(
        (edge.rotation_weight for edge in pose_problem.edges), dtype=np.float64
    )

    acceleration_first = np.fromiter(
        (pose_problem.node_vertex[factor.first_node] for factor in problem.acceleration_factors),
        dtype=np.int64,
    )
    acceleration_second = np.fromiter(
        (pose_problem.node_vertex[factor.second_node] for factor in problem.acceleration_factors),
        dtype=np.int64,
    )
    acceleration_third = np.fromiter(
        (pose_problem.node_vertex[factor.third_node] for factor in problem.acceleration_factors),
        dtype=np.int64,
    )
    acceleration_measurement = np.vstack(
        [factor.delta_velocity for factor in problem.acceleration_factors]
    )
    acceleration_first_duration = np.fromiter(
        (factor.first_duration for factor in problem.acceleration_factors),
        dtype=np.float64,
    )
    acceleration_second_duration = np.fromiter(
        (factor.second_duration for factor in problem.acceleration_factors),
        dtype=np.float64,
    )
    acceleration_factor_weight = acceleration_weight / np.fromiter(
        (factor.loss_duration for factor in problem.acceleration_factors),
        dtype=np.float64,
    )

    rotation_first = np.fromiter(
        (pose_problem.node_vertex[factor.first_node] for factor in problem.rotation_factors),
        dtype=np.int64,
    )
    rotation_second = np.fromiter(
        (pose_problem.node_vertex[factor.second_node] for factor in problem.rotation_factors),
        dtype=np.int64,
    )
    rotation_measurement_inverse = Rotation.from_quat(
        np.vstack(
            [factor.delta_rotation_xyzw for factor in problem.rotation_factors]
        )
    ).inv()
    rotation_factor_weight = rotation_weight / np.fromiter(
        (factor.duration for factor in problem.rotation_factors), dtype=np.float64
    )

    def quaternion_vector_residual(rotations: Rotation) -> np.ndarray:
        quaternions = rotations.as_quat()
        signs = np.where(quaternions[:, 3:4] < 0.0, -1.0, 1.0)
        return 2.0 * quaternions[:, :3] * signs

    def fun(parameters: np.ndarray) -> np.ndarray:
        translations, rotations, gravity, imu_rotation = unpack(parameters)

        source_rotation = rotations[graph_source]
        predicted_graph_translation = source_rotation.inv().apply(
            translations[graph_target] - translations[graph_source]
        )
        predicted_graph_rotation = source_rotation.inv() * rotations[graph_target]
        graph_residual = np.column_stack(
            (
                (predicted_graph_translation - graph_translation)
                * graph_translation_weight[:, None],
                quaternion_vector_residual(
                    graph_rotation_inverse * predicted_graph_rotation
                )
                * graph_rotation_weight[:, None],
            )
        ).reshape(-1)

        measured_world_delta_velocity = rotations[acceleration_second].apply(
            imu_rotation.apply(acceleration_measurement)
        )
        kinematic_delta_velocity = (
            (translations[acceleration_third] - translations[acceleration_second])
            / acceleration_second_duration[:, None]
            - (translations[acceleration_second] - translations[acceleration_first])
            / acceleration_first_duration[:, None]
        )
        kinematic_delta_velocity[:, 2] += gravity * 0.5 * (
            acceleration_first_duration + acceleration_second_duration
        )
        acceleration_residual = (
            measured_world_delta_velocity - kinematic_delta_velocity
        ) * acceleration_factor_weight[:, None]

        predicted_delta_rotation = (
            rotations[rotation_first].inv() * rotations[rotation_second]
        )
        rotation_error = (
            imu_rotation
            * rotation_measurement_inverse
            * imu_rotation.inv()
            * predicted_delta_rotation
        )
        rotation_residual = quaternion_vector_residual(rotation_error) * (
            rotation_factor_weight[:, None]
        )

        calibration_residual = np.concatenate(
            (
                np.array(
                    [
                        (gravity - calibration_options.gravity_magnitude)
                        * calibration_options.gravity_prior_weight
                    ]
                ),
                imu_rotation.as_rotvec()
                * calibration_options.imu_orientation_prior_weight,
            )
        )
        return np.concatenate(
            (
                graph_residual,
                acceleration_residual.reshape(-1),
                rotation_residual.reshape(-1),
                calibration_residual,
            )
        )

    graph_rows = 6 * len(pose_problem.edges)
    acceleration_rows = 3 * len(problem.acceleration_factors)
    rotation_rows = 3 * len(problem.rotation_factors)
    sparsity = lil_matrix(
        (graph_rows + acceleration_rows + rotation_rows + 4, parameter_count),
        dtype=np.int8,
    )
    for edge_index, edge in enumerate(pose_problem.edges):
        rows = slice(6 * edge_index, 6 * edge_index + 6)
        for vertex in (edge.source, edge.target):
            if vertex in pose_offset:
                offset = pose_offset[vertex]
                sparsity[rows, offset : offset + 6] = 1
            elif vertex == gauge_tilt_vertex:
                sparsity[rows, gauge_tilt_offset : gauge_tilt_offset + 2] = 1
    for factor_index, factor in enumerate(problem.acceleration_factors):
        row = graph_rows + 3 * factor_index
        rows = slice(row, row + 3)
        for node_id in (factor.first_node, factor.second_node, factor.third_node):
            vertex = pose_problem.node_vertex[node_id]
            if vertex in pose_offset:
                offset = pose_offset[vertex]
                sparsity[rows, offset : offset + 6] = 1
            elif vertex == gauge_tilt_vertex:
                sparsity[rows, gauge_tilt_offset : gauge_tilt_offset + 2] = 1
        sparsity[rows, gravity_offset] = 1
        sparsity[rows, orientation_offset : orientation_offset + 3] = 1
    rotation_row_base = graph_rows + acceleration_rows
    for factor_index, factor in enumerate(problem.rotation_factors):
        row = rotation_row_base + 3 * factor_index
        rows = slice(row, row + 3)
        for node_id in (factor.first_node, factor.second_node):
            vertex = pose_problem.node_vertex[node_id]
            if vertex in pose_offset:
                offset = pose_offset[vertex]
                sparsity[rows, offset + 3 : offset + 6] = 1
            elif vertex == gauge_tilt_vertex:
                sparsity[rows, gauge_tilt_offset : gauge_tilt_offset + 2] = 1
        sparsity[rows, orientation_offset : orientation_offset + 3] = 1
    prior_row = graph_rows + acceleration_rows + rotation_rows
    sparsity[prior_row, gravity_offset] = 1
    sparsity[prior_row + 1 : prior_row + 4, orientation_offset : orientation_offset + 3] = 1

    initial_residual = fun(x0)
    lower = np.full(parameter_count, -np.inf)
    upper = np.full(parameter_count, np.inf)
    lower[gravity_offset] = 0.0
    result = least_squares(
        fun,
        x0,
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss="linear",
        max_nfev=max_iterations,
        x_scale="jac",
        bounds=(lower, upper),
    )
    translations, rotations, gravity, imu_rotation = unpack(result.x)
    final_residual = fun(result.x)
    poses = tuple(
        Rigid3(translation, quaternion)
        for translation, quaternion in zip(translations, rotations.as_quat())
    )
    calibration = ImuCalibration(
        gravity_magnitude=gravity,
        imu_from_tracking=Rigid3(
            problem.calibration.imu_from_tracking.translation,
            imu_rotation.as_quat(),
        ),
        linear_acceleration_bias=problem.calibration.linear_acceleration_bias,
        linear_acceleration_scaling=problem.calibration.linear_acceleration_scaling,
        linear_acceleration_cross_axis=problem.calibration.linear_acceleration_cross_axis,
        angular_velocity_bias=problem.calibration.angular_velocity_bias,
        angular_velocity_scaling=problem.calibration.angular_velocity_scaling,
        angular_velocity_cross_axis=problem.calibration.angular_velocity_cross_axis,
    )
    return FastImuPoseGraphResult(
        poses,
        0.5 * float(initial_residual @ initial_residual),
        0.5 * float(final_residual @ final_residual),
        int(result.nfev),
        bool(result.success),
        str(result.message),
        calibration,
    )


def optimize_fast_imu_pose_graph_ceres(
    problem: FastImuPoseGraphProblem,
    solver_binary: str | Path,
    work_directory: str | Path,
    *,
    max_iterations: int = 200,
    num_threads: int = 7,
) -> FastImuPoseGraphResult:
    """Run the native Stage1 solver with the installed Ceres graph layout.

    The binary interchange is deliberately small and versioned.  It contains
    only clean-room graph inputs and outputs; the vendor result is never
    supplied to the worker.  Files live in the caller-provided work directory
    so portable regressions do not leak intermediates into ``code/`` or
    ``/tmp``.
    """

    solver = Path(solver_binary).resolve()
    if not solver.is_file():
        raise FileNotFoundError(f"Stage1 Ceres solver does not exist: {solver}")
    if max_iterations < 1 or num_threads < 1:
        raise ValueError("iterations and solver threads must be positive")
    work = Path(work_directory).resolve()
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "stage1_ceres_problem.bin"
    output_path = work / "stage1_ceres_result.bin"

    pose_problem = problem.pose_graph
    fixed_vertices = tuple(pose_problem.fixed_vertices)
    if len(fixed_vertices) != 1:
        raise ValueError("native Stage1 solver expects one fixed gauge vertex")
    header = struct.pack(
        "<8s9I5d",
        b"NVSG1CR1",
        1,
        len(pose_problem.initial_poses),
        len(problem.node_vertices),
        len(pose_problem.edges),
        len(problem.acceleration_factors),
        len(problem.rotation_factors),
        fixed_vertices[0],
        max_iterations,
        num_threads,
        problem.calibration.gravity_magnitude,
        *problem.calibration.imu_from_tracking.quaternion_xyzw,
    )
    with input_path.open("wb") as stream:
        stream.write(header)
        for pose in pose_problem.initial_poses:
            stream.write(
                # The installed backend passes the stored Eigen coefficients
                # directly to Ceres.  Reconstructing a SciPy Rotation here
                # silently renormalizes them and changes the low bits of both
                # the initial residual and the optimized trajectory.
                struct.pack("<7d", *pose.translation, *pose.quaternion_xyzw)
            )
        for edge in pose_problem.edges:
            stream.write(
                struct.pack(
                    "<3I9d",
                    edge.source,
                    edge.target,
                    int(edge.kind == "loop"),
                    *edge.source_from_target.translation,
                    *edge.source_from_target.quaternion_xyzw,
                    edge.translation_weight,
                    edge.rotation_weight,
                )
            )
        for factor in problem.acceleration_factors:
            stream.write(
                struct.pack(
                    "<3I6d",
                    pose_problem.node_vertex[factor.first_node],
                    pose_problem.node_vertex[factor.second_node],
                    pose_problem.node_vertex[factor.third_node],
                    *factor.delta_velocity,
                    factor.first_duration,
                    factor.second_duration,
                    factor.loss_duration,
                )
            )
        for factor in problem.rotation_factors:
            stream.write(
                struct.pack(
                    "<2I5d",
                    pose_problem.node_vertex[factor.first_node],
                    pose_problem.node_vertex[factor.second_node],
                    *factor.delta_rotation_xyzw,
                    factor.duration,
                )
            )

    completed = subprocess.run(
        [str(solver), str(input_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_native_solver_environment(solver),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Stage1 Ceres solver failed with code {completed.returncode}: {detail}"
        )
    raw = output_path.read_bytes()
    prefix_format = "<8s4I7d"
    prefix_size = struct.calcsize(prefix_format)
    if len(raw) < prefix_size:
        raise ValueError("truncated Stage1 Ceres result")
    (
        magic,
        version,
        pose_count,
        iterations,
        success,
        initial_cost,
        final_cost,
        gravity,
        *imu_quaternion,
    ) = struct.unpack_from(prefix_format, raw)
    if magic != b"NVSG1RS1" or version != 1:
        raise ValueError("unsupported Stage1 Ceres result schema")
    if pose_count != len(pose_problem.initial_poses):
        raise ValueError("Stage1 Ceres result pose count changed")
    pose_format = "<7d"
    pose_size = struct.calcsize(pose_format)
    expected_size = prefix_size + pose_count * pose_size
    if len(raw) != expected_size:
        raise ValueError("Stage1 Ceres result has an invalid byte length")
    poses = []
    offset = prefix_size
    for _ in range(pose_count):
        values = struct.unpack_from(pose_format, raw, offset)
        poses.append(Rigid3(np.array(values[:3]), np.array(values[3:])))
        offset += pose_size
    calibration = ImuCalibration(
        gravity_magnitude=gravity,
        imu_from_tracking=Rigid3(
            problem.calibration.imu_from_tracking.translation,
            np.asarray(imu_quaternion),
        ),
        linear_acceleration_bias=problem.calibration.linear_acceleration_bias,
        linear_acceleration_scaling=problem.calibration.linear_acceleration_scaling,
        linear_acceleration_cross_axis=problem.calibration.linear_acceleration_cross_axis,
        angular_velocity_bias=problem.calibration.angular_velocity_bias,
        angular_velocity_scaling=problem.calibration.angular_velocity_scaling,
        angular_velocity_cross_axis=problem.calibration.angular_velocity_cross_axis,
    )
    message = completed.stderr.strip() or completed.stdout.strip()
    return FastImuPoseGraphResult(
        tuple(poses),
        initial_cost,
        final_cost,
        iterations,
        bool(success),
        message,
        calibration,
    )


def _online_prefix_submaps(
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


def _online_matching_submap(
    node_index: int, submaps: Sequence[Submap]
) -> Submap:
    memberships = [submap for submap in submaps if node_index in submap.node_indices]
    if memberships:
        return min(memberships, key=lambda submap: submap.submap_id.index)
    # A replacement submap is created on the threshold node, but insertion
    # begins on the following retained node. The older active map still owns
    # matching at that boundary.
    older = [
        submap
        for submap in submaps
        if submap.node_indices and submap.node_indices[0] <= node_index
    ]
    if not older:
        raise ValueError(f"node {node_index} has no online matching submap")
    return min(older, key=lambda submap: submap.submap_id.index)


def _online_imu_sample_window(
    samples: Sequence[ImuSample], first_timestamp_ns: int, last_timestamp_ns: int
) -> tuple[ImuSample, ...]:
    timestamps = tuple(sample.timestamp_ns for sample in samples)
    first = max(0, bisect_right(timestamps, first_timestamp_ns) - 1)
    last = min(len(samples), bisect_left(timestamps, last_timestamp_ns) + 1)
    return tuple(samples[first:last])


def _extend_online_fast_initial_poses(
    nodes: Sequence[TrajectoryNode],
    submaps: Sequence[Submap],
    previous: OnlineFastPoseGraphSnapshot | None,
) -> tuple[tuple[Rigid3, ...], ImuCalibration]:
    if previous is None:
        return (
            tuple(node.local_pose for node in nodes)
            + tuple(submap.local_pose for submap in submaps),
            ImuCalibration(gravity_magnitude=9.807232),
        )

    previous_submap_pose = {
        submap.submap_id: previous.result.poses[previous.node_count + index]
        for index, submap in enumerate(previous.submaps)
    }
    poses: list[Rigid3] = list(previous.result.poses[: previous.node_count])
    for node in nodes[previous.node_count :]:
        matching = _online_matching_submap(node.node_id.index, submaps)
        matching_global = previous_submap_pose.get(matching.submap_id)
        if matching_global is None:
            bridge = min(
                (
                    candidate
                    for candidate in submaps
                    if candidate.submap_id in previous_submap_pose
                    and node.node_id.index in candidate.node_indices
                ),
                key=lambda candidate: candidate.submap_id.index,
            )
            matching_global = _raw_compose_pose(
                _raw_compose_pose(
                    previous_submap_pose[bridge.submap_id],
                    _raw_inverse_pose(bridge.local_pose),
                ),
                matching.local_pose,
            )
        poses.append(
            _raw_compose_pose(
                _raw_compose_pose(
                    matching_global, _raw_inverse_pose(matching.local_pose)
                ),
                node.local_pose,
            )
        )

    for submap in submaps:
        known = previous_submap_pose.get(submap.submap_id)
        if known is not None:
            poses.append(known)
            continue
        bridge = max(previous.submaps, key=lambda candidate: candidate.submap_id.index)
        poses.append(
            _raw_compose_pose(
                _raw_compose_pose(
                    previous_submap_pose[bridge.submap_id],
                    _raw_inverse_pose(bridge.local_pose),
                ),
                submap.local_pose,
            )
        )
    return tuple(poses), previous.result.calibration


def replay_online_fast_pose_graph(
    nodes: Sequence[TrajectoryNode],
    submaps: Sequence[Submap],
    samples: Sequence[ImuSample],
    solver_binary: str | Path,
    work_directory: str | Path,
    *,
    optimize_every_n_nodes: int = 321,
    max_iterations: int = 10,
    num_threads: int = 7,
    stop_after_node: int | None = None,
) -> tuple[OnlineFastPoseGraphSnapshot, ...]:
    """Replay the installed periodic Fast-IMU optimizations and warm starts.

    The G11 binary's configured value 321 produces periodic solves at
    retained-node counts 321, 642, ... . Loop matching starts from the latest
    of these snapshots. The distinct finish solve is performed only after
    pending loop constraints and the trailing in-memory submap are available;
    use :func:`finish_online_fast_pose_graph` for that stage.
    """

    if optimize_every_n_nodes < 1:
        raise ValueError("online optimization period must be positive")
    ordered_nodes = tuple(
        sorted(nodes, key=lambda node: (node.node_id.trajectory, node.node_id.index))
    )
    ordered_submaps = tuple(
        sorted(
            submaps,
            key=lambda submap: (
                submap.submap_id.trajectory,
                submap.submap_id.index,
            ),
        )
    )
    maximum_count = len(ordered_nodes)
    if stop_after_node is not None:
        maximum_count = min(maximum_count, stop_after_node + 1)
    snapshots: list[OnlineFastPoseGraphSnapshot] = []
    work = Path(work_directory)
    solve_counts = list(
        range(optimize_every_n_nodes, maximum_count + 1, optimize_every_n_nodes)
    )
    for count in solve_counts:
        prefix_nodes = ordered_nodes[:count]
        prefix_submaps = _online_prefix_submaps(ordered_submaps, prefix_nodes)
        graph = build_pose_graph(prefix_nodes, prefix_submaps, ())
        initial_poses, calibration = _extend_online_fast_initial_poses(
            prefix_nodes,
            prefix_submaps,
            snapshots[-1] if snapshots else None,
        )
        graph = replace(graph, initial_poses=initial_poses)
        sample_window = _online_imu_sample_window(
            samples,
            prefix_nodes[0].timestamp_ns,
            prefix_nodes[-1].timestamp_ns,
        )
        problem = build_fast_imu_pose_graph(
            graph, prefix_nodes, sample_window, calibration
        )
        result = optimize_fast_imu_pose_graph_ceres(
            problem,
            solver_binary,
            work / f"solve_{count:05d}",
            max_iterations=max_iterations,
            num_threads=num_threads,
        )
        snapshots.append(
            OnlineFastPoseGraphSnapshot(count, prefix_submaps, result)
        )
    return tuple(snapshots)


def finish_online_fast_pose_graph(
    nodes: Sequence[TrajectoryNode],
    submaps: Sequence[Submap],
    samples: Sequence[ImuSample],
    loops: Sequence[LoopConstraint],
    periodic_snapshots: Sequence[OnlineFastPoseGraphSnapshot],
    solver_binary: str | Path,
    work_directory: str | Path,
    *,
    max_iterations: int = 200,
    num_threads: int = 7,
) -> OnlineFastPoseGraphSnapshot:
    """Run ``FinishTrajectory`` with loops and all in-memory submaps.

    The trailing submap is deliberately part of this Ceres problem even
    though it is omitted from the serialized submap archive afterwards.
    """

    ordered_nodes = tuple(
        sorted(nodes, key=lambda node: (node.node_id.trajectory, node.node_id.index))
    )
    if not ordered_nodes:
        raise ValueError("online finish needs trajectory nodes")
    ordered_submaps = tuple(
        sorted(
            submaps,
            key=lambda submap: (
                submap.submap_id.trajectory,
                submap.submap_id.index,
            ),
        )
    )
    if not ordered_submaps:
        raise ValueError("online finish needs submaps")
    final_submaps = _online_prefix_submaps(ordered_submaps, ordered_nodes)
    graph = build_pose_graph(ordered_nodes, final_submaps, loops)
    initial_poses, calibration = _extend_online_fast_initial_poses(
        ordered_nodes,
        final_submaps,
        periodic_snapshots[-1] if periodic_snapshots else None,
    )
    graph = replace(graph, initial_poses=initial_poses)
    sample_window = _online_imu_sample_window(
        samples,
        ordered_nodes[0].timestamp_ns,
        ordered_nodes[-1].timestamp_ns,
    )
    problem = build_fast_imu_pose_graph(
        graph, ordered_nodes, sample_window, calibration
    )
    result = optimize_fast_imu_pose_graph_ceres(
        problem,
        solver_binary,
        Path(work_directory) / f"finish_{len(ordered_nodes):05d}",
        max_iterations=max_iterations,
        num_threads=num_threads,
    )
    return OnlineFastPoseGraphSnapshot(len(ordered_nodes), final_submaps, result)


def online_fast_loop_initial_pose(
    snapshots: Sequence[OnlineFastPoseGraphSnapshot],
    nodes: Sequence[TrajectoryNode],
    submaps: Sequence[Submap],
    target_submap_id: NodeId,
    node_id: NodeId,
) -> Rigid3:
    """Return the online global-state seed for a loop matcher pair."""

    node_by_id = {node.node_id: node for node in nodes}
    submap_by_id = {submap.submap_id: submap for submap in submaps}
    node = node_by_id[node_id]
    target = submap_by_id[target_submap_id]
    if not snapshots:
        return _raw_relative_pose(target.local_pose, node.local_pose)
    # Constraint jobs are created while scans are still arriving. The latest
    # periodic solve is therefore the seed; the finish solve runs only after
    # accepted jobs have supplied their loop edges.
    snapshot = snapshots[-1]
    global_submaps = {
        submap.submap_id: snapshot.result.poses[snapshot.node_count + index]
        for index, submap in enumerate(snapshot.submaps)
    }
    target_global = global_submaps[target_submap_id]
    if node_id.index < snapshot.node_count:
        node_global = snapshot.result.poses[node_id.index]
    else:
        current_submaps = _online_prefix_submaps(
            submaps, nodes[: node_id.index + 1]
        )
        matching = _online_matching_submap(node_id.index, current_submaps)
        matching_global = global_submaps[matching.submap_id]
        node_global = _raw_compose_pose(
            _raw_compose_pose(
                matching_global, _raw_inverse_pose(matching.local_pose)
            ),
            node.local_pose,
        )
    return _raw_relative_pose(target_global, node_global)


def optimize_imu_pose_graph_ceres(
    problem: ImuPoseGraphProblem,
    solver_binary: str | Path,
    work_directory: str | Path,
    *,
    max_iterations: int = 200,
    num_threads: int = 7,
    imu_rotation_weight: float = 1000.0,
    imu_velocity_weight: float = 1000.0,
    imu_position_weight: float = 1000.0,
    calibration_options: ImuCalibrationOptions = ImuCalibrationOptions(),
) -> ImuPoseGraphResult:
    """Run the native joint-calibration Stage2 Ceres worker.

    The interchange contains only the clean Stage1 state, graph constraints,
    raw IMU samples and calibration priors.  Endpoint interpolation is encoded
    as raw-sample references so the worker preserves Stage2's exact
    correction -> interpolation -> half-step integration order.  A final
    vendor state is neither accepted nor serialized by this function.
    """

    solver = Path(solver_binary).resolve()
    if not solver.is_file():
        raise FileNotFoundError(f"Stage2 Ceres solver does not exist: {solver}")
    weights = (imu_rotation_weight, imu_velocity_weight, imu_position_weight)
    if max_iterations < 1 or num_threads < 1 or any(weight <= 0 for weight in weights):
        raise ValueError("iterations, threads and IMU weights must be positive")
    if not problem.samples:
        raise ValueError("native Stage2 joint calibration requires raw IMU samples")

    pose_problem = problem.pose_graph
    fixed_vertices = tuple(pose_problem.fixed_vertices)
    if len(fixed_vertices) != 1:
        raise ValueError("native Stage2 solver expects one fixed gauge vertex")
    if len(problem.node_vertices) != len(problem.initial_velocities):
        raise ValueError("Stage2 node vertices and velocities have different counts")

    work = Path(work_directory).resolve()
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "stage2_ceres_problem.bin"
    output_path = work / "stage2_ceres_result.bin"

    sample_times = tuple(sample.timestamp_ns for sample in problem.samples)
    if len(sample_times) < 2 or any(
        second <= first for first, second in zip(sample_times, sample_times[1:])
    ):
        raise ValueError("Stage2 IMU samples must be strictly increasing")
    node_order = {vertex: index for index, vertex in enumerate(problem.node_vertices)}

    factor_points: list[tuple[tuple[int, int, float, float], ...]] = []
    for factor in problem.preintegrations:
        first_inside = bisect_right(sample_times, factor.source_timestamp_ns)
        last_inside = bisect_left(sample_times, factor.target_timestamp_ns)
        times = [factor.source_timestamp_ns]
        times.extend(sample_times[first_inside:last_inside])
        times.append(factor.target_timestamp_ns)
        points: list[tuple[int, int, float, float]] = []
        for point_index, timestamp_ns in enumerate(times):
            sample_index = bisect_left(sample_times, timestamp_ns)
            if (
                sample_index < len(sample_times)
                and sample_times[sample_index] == timestamp_ns
            ):
                before = after = sample_index
                alpha = 0.0
            else:
                before = sample_index - 1
                after = sample_index
                alpha = float(timestamp_ns - sample_times[before]) / float(
                    sample_times[after] - sample_times[before]
                )
            dt_to_next = (
                float(times[point_index + 1] - timestamp_ns) / 1.0e9
                if point_index + 1 < len(times)
                else 0.0
            )
            points.append((before, after, alpha, dt_to_next))
        factor_points.append(tuple(points))

    calibration = problem.calibration
    header_values = (
        imu_rotation_weight,
        imu_velocity_weight,
        imu_position_weight,
        calibration.gravity_magnitude,
        *calibration.imu_from_tracking.quaternion_xyzw,
        *calibration.linear_acceleration_bias,
        *calibration.linear_acceleration_scaling,
        *calibration.linear_acceleration_cross_axis,
        *calibration.angular_velocity_bias,
        *calibration.angular_velocity_scaling,
        *calibration.angular_velocity_cross_axis,
        calibration_options.gravity_magnitude,
        calibration_options.gravity_prior_weight,
        calibration_options.imu_orientation_prior_weight,
        *calibration_options.linear_acceleration_bias_prior_weight,
        *calibration_options.linear_acceleration_scaling_prior_weight,
        *calibration_options.angular_velocity_bias_prior_weight,
        *calibration_options.angular_velocity_scaling_prior_weight,
    )
    if len(header_values) != 47:
        raise AssertionError("Stage2 Ceres schema header has the wrong field count")

    with input_path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8s10I",
                b"NVSG2CR1",
                1,
                len(pose_problem.initial_poses),
                len(problem.node_vertices),
                len(pose_problem.edges),
                len(problem.samples),
                len(problem.preintegrations),
                fixed_vertices[0],
                max_iterations,
                num_threads,
                0,
            )
        )
        stream.write(struct.pack("<47d", *header_values))
        for pose in pose_problem.initial_poses:
            stream.write(
                # Stage2 starts from the serialized optimization-state
                # coefficients. The installed factor does not replace those
                # coefficients with a freshly normalized Rotation object
                # before its first Ceres evaluation.
                struct.pack("<7d", *pose.translation, *pose.quaternion_xyzw)
            )
        for vertex, velocity in zip(problem.node_vertices, problem.initial_velocities):
            stream.write(struct.pack("<I3d", vertex, *velocity))
        for edge in pose_problem.edges:
            stream.write(
                struct.pack(
                    "<3I9d",
                    edge.source,
                    edge.target,
                    int(edge.kind == "loop"),
                    *edge.source_from_target.translation,
                    *edge.source_from_target.quaternion_xyzw,
                    edge.translation_weight,
                    edge.rotation_weight,
                )
            )
        for sample in problem.samples:
            stream.write(
                struct.pack(
                    "<6d", *sample.linear_acceleration, *sample.angular_velocity
                )
            )
        for factor, points in zip(problem.preintegrations, factor_points):
            source_vertex = pose_problem.node_vertex[factor.source_node]
            target_vertex = pose_problem.node_vertex[factor.target_node]
            stream.write(
                struct.pack(
                    "<5Id",
                    source_vertex,
                    target_vertex,
                    node_order[source_vertex],
                    node_order[target_vertex],
                    len(points),
                    factor.delta_t,
                )
            )
            for before, after, alpha, dt_to_next in points:
                stream.write(
                    struct.pack("<2I2d", before, after, alpha, dt_to_next)
                )

    completed = subprocess.run(
        [str(solver), str(input_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_native_solver_environment(solver),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Stage2 Ceres solver failed with code {completed.returncode}: {detail}"
        )

    raw = output_path.read_bytes()
    prefix_format = "<8s5I19d"
    prefix_size = struct.calcsize(prefix_format)
    if len(raw) < prefix_size:
        raise ValueError("truncated Stage2 Ceres result")
    (
        magic,
        version,
        pose_count,
        velocity_count,
        iterations,
        success,
        initial_cost,
        final_cost,
        gravity,
        *calibration_values,
    ) = struct.unpack_from(prefix_format, raw)
    if magic != b"NVSG2RS1" or version != 1:
        raise ValueError("unsupported Stage2 Ceres result schema")
    if pose_count != len(pose_problem.initial_poses):
        raise ValueError("Stage2 Ceres result pose count changed")
    if velocity_count != len(problem.node_vertices):
        raise ValueError("Stage2 Ceres result velocity count changed")
    pose_size = struct.calcsize("<7d")
    velocity_size = struct.calcsize("<3d")
    expected_size = prefix_size + pose_count * pose_size + velocity_count * velocity_size
    if len(raw) != expected_size:
        raise ValueError("Stage2 Ceres result has an invalid byte length")

    poses: list[Rigid3] = []
    offset = prefix_size
    for _ in range(pose_count):
        values = struct.unpack_from("<7d", raw, offset)
        poses.append(Rigid3(np.asarray(values[:3]), np.asarray(values[3:])))
        offset += pose_size
    velocities = np.empty((velocity_count, 3), dtype=np.float64)
    for index in range(velocity_count):
        velocities[index] = struct.unpack_from("<3d", raw, offset)
        offset += velocity_size

    imu_quaternion = np.asarray(calibration_values[0:4])
    linear_acceleration_bias = np.asarray(calibration_values[4:7])
    linear_acceleration_scaling = np.asarray(calibration_values[7:10])
    angular_velocity_bias = np.asarray(calibration_values[10:13])
    angular_velocity_scaling = np.asarray(calibration_values[13:16])
    final_calibration = ImuCalibration(
        gravity_magnitude=gravity,
        imu_from_tracking=Rigid3(
            calibration.imu_from_tracking.translation, imu_quaternion
        ),
        linear_acceleration_bias=linear_acceleration_bias,
        linear_acceleration_scaling=linear_acceleration_scaling,
        linear_acceleration_cross_axis=calibration.linear_acceleration_cross_axis,
        angular_velocity_bias=angular_velocity_bias,
        angular_velocity_scaling=angular_velocity_scaling,
        angular_velocity_cross_axis=calibration.angular_velocity_cross_axis,
    )
    message = completed.stderr.strip() or completed.stdout.strip()
    return ImuPoseGraphResult(
        tuple(poses),
        velocities,
        initial_cost,
        final_cost,
        iterations,
        bool(success),
        message,
        final_calibration,
    )


def optimize_imu_pose_graph(
    problem: ImuPoseGraphProblem,
    *,
    max_iterations: int = 200,
    imu_rotation_weight: float = 1000.0,
    imu_velocity_weight: float = 1000.0,
    imu_position_weight: float = 1000.0,
    robust_loss: str = "linear",
    robust_scale: float = 1.0,
    calibrate_imu_intrinsics: bool = False,
    calibration_options: ImuCalibrationOptions = ImuCalibrationOptions(),
) -> ImuPoseGraphResult:
    """Jointly optimize node/submap poses and node velocities with IMU edges.

    The residual layout is the installed Exact factor's ``(position, rotation,
    velocity)`` triplet.  Every group is weighted by ``1000 / sqrt(dt)``;
    position and rotation stay in the source node frame while velocity is
    expressed in the world frame.  Those conventions were verified against a
    single-thread capture of the same 15 parameter blocks and nine outputs.

    When ``calibrate_imu_intrinsics`` is enabled, the shared gravity constant,
    IMU orientation, acceleration bias/scaling and gyroscope bias/scaling are
    optimized with the exact installed priors.  Translation and both six-term
    cross-axis matrices remain fixed because the reference configuration
    disables those parameter blocks.
    """

    weights = (imu_rotation_weight, imu_velocity_weight, imu_position_weight)
    if max_iterations < 1 or any(weight <= 0 for weight in weights):
        raise ValueError("iterations and IMU weights must be positive")

    pose_problem = problem.pose_graph
    fixed = set(pose_problem.fixed_vertices)
    # The original IMU graph fixes the global translation and yaw gauge while
    # allowing gravity to re-estimate roll/pitch.  Its first retained submap
    # has exactly zero translation change but a measurable tilt correction.
    gauge_tilt_vertex = min(fixed)
    variable_pose_vertices = [
        index for index in range(len(pose_problem.initial_poses)) if index not in fixed
    ]
    pose_offset = {
        vertex: 6 * index for index, vertex in enumerate(variable_pose_vertices)
    }
    gauge_tilt_offset = 6 * len(variable_pose_vertices)
    velocity_base = gauge_tilt_offset + 2
    velocity_offset = {
        vertex: velocity_base + 3 * index
        for index, vertex in enumerate(problem.node_vertices)
    }
    calibration_base = velocity_base + 3 * len(problem.node_vertices)
    calibration_offsets: dict[str, int] = {}
    parameter_count = calibration_base
    if calibrate_imu_intrinsics:
        if not problem.samples:
            raise ValueError("live IMU calibration requires the retained IMU samples")
        for name, size in (
            ("gravity", 1),
            ("imu_orientation", 3),
            ("linear_acceleration_bias", 3),
            ("linear_acceleration_scaling", 3),
            ("angular_velocity_bias", 3),
            ("angular_velocity_scaling", 3),
        ):
            calibration_offsets[name] = parameter_count
            parameter_count += size
    initial_translations = np.vstack(
        [pose.translation for pose in pose_problem.initial_poses]
    )
    initial_quaternions = np.vstack(
        [pose.quaternion_xyzw for pose in pose_problem.initial_poses]
    )
    variable_pose_vertex_array = np.asarray(variable_pose_vertices, dtype=np.int64)
    variable_initial_rotations = Rotation.from_quat(
        initial_quaternions[variable_pose_vertex_array]
    )
    gauge_initial_rotation = Rotation.from_quat(
        initial_quaternions[gauge_tilt_vertex]
    )
    x0 = np.empty(parameter_count, dtype=np.float64)
    for vertex, offset in pose_offset.items():
        x0[offset : offset + 3] = pose_problem.initial_poses[vertex].translation
        x0[offset + 3 : offset + 6] = 0.0
    x0[gauge_tilt_offset : gauge_tilt_offset + 2] = 0.0
    for index, vertex in enumerate(problem.node_vertices):
        offset = velocity_offset[vertex]
        x0[offset : offset + 3] = problem.initial_velocities[index]
    if calibrate_imu_intrinsics:
        initial_calibration = problem.calibration
        x0[calibration_offsets["gravity"]] = initial_calibration.gravity_magnitude
        x0[
            calibration_offsets["imu_orientation"] :
            calibration_offsets["imu_orientation"] + 3
        ] = 0.0
        for name in (
            "linear_acceleration_bias",
            "linear_acceleration_scaling",
            "angular_velocity_bias",
            "angular_velocity_scaling",
        ):
            offset = calibration_offsets[name]
            x0[offset : offset + 3] = getattr(initial_calibration, name)

    initial_imu_rotation = problem.calibration.imu_from_tracking.rotation

    def unpack_calibration(parameters: np.ndarray) -> ImuCalibration:
        if not calibrate_imu_intrinsics:
            return problem.calibration
        orientation_offset = calibration_offsets["imu_orientation"]
        imu_rotation = (
            Rotation.from_rotvec(parameters[orientation_offset : orientation_offset + 3])
            * initial_imu_rotation
        )
        return ImuCalibration(
            gravity_magnitude=float(parameters[calibration_offsets["gravity"]]),
            imu_from_tracking=Rigid3(
                problem.calibration.imu_from_tracking.translation,
                imu_rotation.as_quat(),
            ),
            linear_acceleration_bias=parameters[
                calibration_offsets["linear_acceleration_bias"] :
                calibration_offsets["linear_acceleration_bias"] + 3
            ],
            linear_acceleration_scaling=parameters[
                calibration_offsets["linear_acceleration_scaling"] :
                calibration_offsets["linear_acceleration_scaling"] + 3
            ],
            linear_acceleration_cross_axis=(
                problem.calibration.linear_acceleration_cross_axis
            ),
            angular_velocity_bias=parameters[
                calibration_offsets["angular_velocity_bias"] :
                calibration_offsets["angular_velocity_bias"] + 3
            ],
            angular_velocity_scaling=parameters[
                calibration_offsets["angular_velocity_scaling"] :
                calibration_offsets["angular_velocity_scaling"] + 3
            ],
            angular_velocity_cross_axis=problem.calibration.angular_velocity_cross_axis,
        )

    def unpack_arrays(
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Rotation, np.ndarray]:
        translations = initial_translations.copy()
        quaternions = initial_quaternions.copy()
        pose_parameters = parameters[:gauge_tilt_offset].reshape((-1, 6))
        translations[variable_pose_vertex_array] = pose_parameters[:, :3]
        quaternions[variable_pose_vertex_array] = (
            Rotation.from_rotvec(pose_parameters[:, 3:])
            * variable_initial_rotations
        ).as_quat()
        gauge_tilt = Rotation.from_rotvec(
            [parameters[gauge_tilt_offset], parameters[gauge_tilt_offset + 1], 0.0]
        )
        quaternions[gauge_tilt_vertex] = (
            gauge_initial_rotation * gauge_tilt
        ).as_quat()
        velocities = parameters[velocity_base:calibration_base].reshape((-1, 3))
        return translations, quaternions, Rotation.from_quat(quaternions), velocities

    node_order = {vertex: index for index, vertex in enumerate(problem.node_vertices)}
    # All graph topology and measurements are constant during optimization.
    # Packing them once lets each numerical-Jacobian evaluation run as a small
    # number of vectorized Rotation/NumPy calls instead of constructing tens of
    # thousands of temporary Rigid3 objects in Python.
    graph_source = np.fromiter(
        (edge.source for edge in pose_problem.edges), dtype=np.int64
    )
    graph_target = np.fromiter(
        (edge.target for edge in pose_problem.edges), dtype=np.int64
    )
    graph_translation = np.vstack(
        [edge.source_from_target.translation for edge in pose_problem.edges]
    )
    graph_rotation_inverse = Rotation.from_quat(
        np.vstack(
            [edge.source_from_target.quaternion_xyzw for edge in pose_problem.edges]
        )
    ).inv()
    graph_translation_weight = np.fromiter(
        (edge.translation_weight for edge in pose_problem.edges), dtype=np.float64
    )
    graph_rotation_weight = np.fromiter(
        (edge.rotation_weight for edge in pose_problem.edges), dtype=np.float64
    )

    imu_source_vertex = np.fromiter(
        (
            pose_problem.node_vertex[edge.source_node]
            for edge in problem.preintegrations
        ),
        dtype=np.int64,
    )
    imu_target_vertex = np.fromiter(
        (
            pose_problem.node_vertex[edge.target_node]
            for edge in problem.preintegrations
        ),
        dtype=np.int64,
    )
    imu_source_velocity = np.fromiter(
        (node_order[vertex] for vertex in imu_source_vertex), dtype=np.int64
    )
    imu_target_velocity = np.fromiter(
        (node_order[vertex] for vertex in imu_target_vertex), dtype=np.int64
    )
    imu_delta_t = np.fromiter(
        (edge.delta_t for edge in problem.preintegrations), dtype=np.float64
    )
    imu_rotation_inverse = Rotation.from_quat(
        np.vstack([edge.delta_rotation_xyzw for edge in problem.preintegrations])
    ).inv()
    imu_delta_velocity = np.vstack(
        [edge.delta_velocity for edge in problem.preintegrations]
    )
    imu_delta_position = np.vstack(
        [edge.delta_position for edge in problem.preintegrations]
    )
    imu_sample_times = tuple(sample.timestamp_ns for sample in problem.samples)
    raw_linear_acceleration = np.vstack(
        [sample.linear_acceleration for sample in problem.samples]
    ) if problem.samples else np.empty((0, 3), dtype=np.float64)
    raw_angular_velocity = np.vstack(
        [sample.angular_velocity for sample in problem.samples]
    ) if problem.samples else np.empty((0, 3), dtype=np.float64)

    def calibrated_imu_measurements(
        calibration: ImuCalibration,
    ) -> tuple[Rotation, np.ndarray, np.ndarray]:
        if not calibrate_imu_intrinsics:
            return imu_rotation_inverse, imu_delta_velocity, imu_delta_position

        def correct_stream(
            values: np.ndarray,
            bias: np.ndarray,
            scaling: np.ndarray,
            cross: np.ndarray,
        ) -> np.ndarray:
            cross_matrix = np.array(
                [
                    [1.0, cross[0], cross[1]],
                    [cross[2], 1.0, cross[3]],
                    [cross[4], cross[5], 1.0],
                ]
            )
            intrinsic = ((values - bias[None, :]) * scaling[None, :])
            return calibration.imu_from_tracking.rotation.apply(
                intrinsic @ cross_matrix.T
            )

        corrected_acceleration = correct_stream(
            raw_linear_acceleration,
            calibration.linear_acceleration_bias,
            calibration.linear_acceleration_scaling,
            calibration.linear_acceleration_cross_axis,
        )
        corrected_angular_velocity = correct_stream(
            raw_angular_velocity,
            calibration.angular_velocity_bias,
            calibration.angular_velocity_scaling,
            calibration.angular_velocity_cross_axis,
        )
        corrected_samples = tuple(
            zip(corrected_acceleration, corrected_angular_velocity)
        )
        edges = tuple(
            _preintegrate_corrected_imu(
                imu_sample_times,
                corrected_samples,
                edge.source_timestamp_ns,
                edge.target_timestamp_ns,
                source_node=edge.source_node,
                target_node=edge.target_node,
            )
            for edge in problem.preintegrations
        )
        return (
            Rotation.from_quat(
                np.vstack([edge.delta_rotation_xyzw for edge in edges])
            ).inv(),
            np.vstack([edge.delta_velocity for edge in edges]),
            np.vstack([edge.delta_position for edge in edges]),
        )

    graph_rows = 6 * len(pose_problem.edges)
    imu_rows = 9 * len(problem.preintegrations)
    calibration_prior_rows = 16 if calibrate_imu_intrinsics else 0

    linear_bias_prior_weight = np.asarray(
        calibration_options.linear_acceleration_bias_prior_weight,
        dtype=np.float64,
    )
    linear_scaling_prior_weight = np.asarray(
        calibration_options.linear_acceleration_scaling_prior_weight,
        dtype=np.float64,
    )
    angular_bias_prior_weight = np.asarray(
        calibration_options.angular_velocity_bias_prior_weight,
        dtype=np.float64,
    )
    angular_scaling_prior_weight = np.asarray(
        calibration_options.angular_velocity_scaling_prior_weight,
        dtype=np.float64,
    )

    def calibration_prior_residual(calibration: ImuCalibration) -> np.ndarray:
        return np.concatenate(
            (
                np.array(
                    [
                        (
                            calibration.gravity_magnitude
                            - calibration_options.gravity_magnitude
                        )
                        * calibration_options.gravity_prior_weight
                    ]
                ),
                calibration.imu_from_tracking.rotation.as_rotvec()
                * calibration_options.imu_orientation_prior_weight,
                calibration.linear_acceleration_bias * linear_bias_prior_weight,
                (calibration.linear_acceleration_scaling - 1.0)
                * linear_scaling_prior_weight,
                calibration.angular_velocity_bias * angular_bias_prior_weight,
                (calibration.angular_velocity_scaling - 1.0)
                * angular_scaling_prior_weight,
            )
        )

    def fun(parameters: np.ndarray) -> np.ndarray:
        translations, _, rotations, velocities = unpack_arrays(parameters)
        calibration = unpack_calibration(parameters)
        gravity = np.array([0.0, 0.0, -calibration.gravity_magnitude])
        (
            current_imu_rotation_inverse,
            current_imu_delta_velocity,
            current_imu_delta_position,
        ) = calibrated_imu_measurements(calibration)

        source_rotation = rotations[graph_source]
        predicted_graph_translation = source_rotation.inv().apply(
            translations[graph_target] - translations[graph_source]
        )
        predicted_graph_rotation = source_rotation.inv() * rotations[graph_target]
        graph_residual = np.column_stack(
            (
                (predicted_graph_translation - graph_translation)
                * graph_translation_weight[:, None],
                (graph_rotation_inverse * predicted_graph_rotation).as_rotvec()
                * graph_rotation_weight[:, None],
            )
        )

        source_rotation = rotations[imu_source_vertex]
        source_velocity = velocities[imu_source_velocity]
        target_velocity = velocities[imu_target_velocity]
        predicted_imu_rotation = source_rotation.inv() * rotations[imu_target_vertex]
        predicted_delta_position = source_rotation.inv().apply(
            translations[imu_target_vertex]
            - translations[imu_source_vertex]
            - source_velocity * imu_delta_t[:, None]
            - 0.5 * gravity[None, :] * np.square(imu_delta_t[:, None])
        )
        normalization = np.reciprocal(np.sqrt(imu_delta_t))[:, None]
        imu_residual = np.column_stack(
            (
                (predicted_delta_position - current_imu_delta_position)
                * (imu_position_weight * normalization),
                (current_imu_rotation_inverse * predicted_imu_rotation).as_rotvec()
                * (imu_rotation_weight * normalization),
                (
                    target_velocity
                    - source_velocity
                    - gravity[None, :] * imu_delta_t[:, None]
                    - source_rotation.apply(current_imu_delta_velocity)
                )
                * (imu_velocity_weight * normalization),
            )
        )
        residual_groups = [graph_residual.reshape(-1), imu_residual.reshape(-1)]
        if calibrate_imu_intrinsics:
            residual_groups.append(calibration_prior_residual(calibration))
        return np.concatenate(residual_groups)

    sparsity = lil_matrix(
        (graph_rows + imu_rows + calibration_prior_rows, len(x0)), dtype=np.int8
    )
    for edge_index, edge in enumerate(pose_problem.edges):
        rows = slice(6 * edge_index, 6 * edge_index + 6)
        for vertex in (edge.source, edge.target):
            if vertex in pose_offset:
                offset = pose_offset[vertex]
                sparsity[rows, offset : offset + 6] = 1
            elif vertex == gauge_tilt_vertex:
                sparsity[rows, gauge_tilt_offset : gauge_tilt_offset + 2] = 1
    for edge_index, edge in enumerate(problem.preintegrations):
        rows = slice(graph_rows + 9 * edge_index, graph_rows + 9 * edge_index + 9)
        for node_id in (edge.source_node, edge.target_node):
            vertex = pose_problem.node_vertex[node_id]
            if vertex in pose_offset:
                offset = pose_offset[vertex]
                sparsity[rows, offset : offset + 6] = 1
            elif vertex == gauge_tilt_vertex:
                sparsity[rows, gauge_tilt_offset : gauge_tilt_offset + 2] = 1
            velocity = velocity_offset[vertex]
            sparsity[rows, velocity : velocity + 3] = 1
    if calibrate_imu_intrinsics:
        imu_row_slice = slice(graph_rows, graph_rows + imu_rows)
        sparsity[imu_row_slice, calibration_base:parameter_count] = 1
        prior_base = graph_rows + imu_rows
        # gravity, orientation and four three-vectors are packed in the same
        # order as ``calibration_prior_residual``.
        sparsity[prior_base, calibration_offsets["gravity"]] = 1
        prior_base += 1
        for name in (
            "imu_orientation",
            "linear_acceleration_bias",
            "linear_acceleration_scaling",
            "angular_velocity_bias",
            "angular_velocity_scaling",
        ):
            offset = calibration_offsets[name]
            sparsity[prior_base : prior_base + 3, offset : offset + 3] = 1
            prior_base += 3

    initial_residual = fun(x0)
    result = least_squares(
        fun,
        x0,
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss=robust_loss,
        f_scale=robust_scale,
        max_nfev=max_iterations,
        x_scale="jac",
    )
    translations, quaternions, _, velocities = unpack_arrays(result.x)
    poses = tuple(
        Rigid3(translation, quaternion)
        for translation, quaternion in zip(translations, quaternions)
    )
    final_residual = fun(result.x)
    final_calibration = unpack_calibration(result.x)
    return ImuPoseGraphResult(
        poses,
        velocities,
        0.5 * float(initial_residual @ initial_residual),
        0.5 * float(final_residual @ final_residual),
        int(result.nfev),
        bool(result.success),
        str(result.message),
        final_calibration,
    )
