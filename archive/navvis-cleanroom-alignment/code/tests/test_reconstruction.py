from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import navvis_recon.surveyor_frontend as surveyor_frontend
from navvis_recon.cloud_builder import BuilderConfig, build_cloud
from navvis_recon.cloud_surface_filter import SurfaceFilterConfig, aggregate_voxels, estimate_multiscale_normals
from navvis_recon.floor_estimator import TraceSample, refined_floor_estimator
from navvis_recon.models import Camera, LaserPoint, Pose
from navvis_recon.pointcloud_coloring import ColoringConfig, View, colorize
from navvis_recon.quality_map import QualityConfig, Ray, mapped_space_quality
from navvis_recon.slam_reconstruction import (
    Trajectory,
    evaluate_relative_trajectory,
    evaluate_trajectory,
    fuse_global_and_local,
)
from navvis_recon.surveyor_frontend import (
    adaptive_first_point_filter,
    CorrelativeResult,
    detect_loop_constraints,
    evaluate_constraint_stability,
    range_measurement_centroid_filter,
    FrontendConfig,
    FrontendNode,
    HybridProbabilityGrid,
    IcpResult,
    OverlappingSubmapBuilder,
    RawConstantVelocityPosePredictor,
    RawImuTracker,
    _compute_rotational_histogram_python,
    compute_rotational_histogram,
    descriptor_distance,
    fast_correlative_scan_match,
    point_to_plane_icp,
    rotational_score_is_acceptable,
    scan_context_descriptor,
)
from navvis_recon.surveyor_slam import (
    ImuCalibration,
    ImuCalibrationOptions,
    ImuSample,
    LoopConstraint,
    NodeId,
    Rigid3,
    Submap,
    TrajectoryNode,
    _integrate_fast_imu,
    build_fast_imu_pose_graph,
    build_imu_pose_graph,
    build_pose_graph,
    optimize_fast_imu_pose_graph,
    optimize_imu_pose_graph,
    optimize_pose_graph,
    preintegrate_imu,
)


def test_native_rotational_histogram_matches_scalar_reference():
    generator = np.random.default_rng(20260828)
    points = generator.normal(size=(12_000, 3)).astype(np.float32)
    points[:, :2] *= np.float32(4.0)
    points[:, 2] *= np.float32(1.5)
    reference = _compute_rotational_histogram_python(points)
    candidate = compute_rotational_histogram(points)
    np.testing.assert_allclose(candidate, reference, rtol=1.0e-6, atol=3.0e-4)


def identity_pose(t, x=0.0):
    return Pose(t, np.array([x, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))


def test_cloud_builder_unskews_with_given_trajectory():
    point = LaserPoint(np.array([0.0, 0.0, 1.0]), 0.5, intensity=1.0)
    cloud = build_cloud([[point]], [identity_pose(0, 0), identity_pose(1, 2)], BuilderConfig())
    np.testing.assert_allclose(cloud[0].xyz, [1, 0, 1])


def test_voxel_aggregation_and_normals():
    points = [
        LaserPoint(np.array([x, y, 1.0]), 0, intensity=1, origin=np.zeros(3))
        for x in np.arange(-0.1, 0.11, 0.02)
        for y in np.arange(-0.1, 0.11, 0.02)
    ]
    cfg = SurfaceFilterConfig(resolution=0.01, normal_min_radius=0.04, normal_max_radius=0.08, normal_levels=3)
    surface = aggregate_voxels(points, cfg.resolution)
    estimate_multiscale_normals(surface, cfg)
    normals = [p.normal for p in surface if p.normal is not None]
    assert len(normals) > 50
    assert np.median([abs(n[2]) for n in normals]) > 0.99


def test_colorizer_depth_tests_and_samples_patch():
    pose = identity_pose(0)
    camera = Camera(21, 21, 10, 10, 10, 10, pose)
    image = np.zeros((21, 21, 3), np.uint8)
    image[:] = [0, 0, 255]  # BGR red
    point = LaserPoint(np.array([0.0, 0.0, 2.0]), 0, normal=np.array([0, 0, -1.0]))
    result = colorize([point], [View(camera, image)], ColoringConfig(exposure="none", patch_radius=1))
    np.testing.assert_array_equal(result[0].rgb, [255, 0, 0])


def test_floor_estimator_separates_vertical_clusters():
    samples = []
    for i in range(80):
        z = 0.04 * np.sin(i) if i < 40 else 3.0 + 0.04 * np.sin(i)
        samples.append(TraceSample(i * 200_000_000, 0, 0, z))
    floors = refined_floor_estimator(samples)
    assert len(floors) == 2
    assert floors[1].center - floors[0].center > 2.8


def test_quality_voxels_count_rays():
    rays = [Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) for _ in range(40)]
    voxels = mapped_space_quality(rays, QualityConfig(grid_resolution=0.25, min_num_rays_per_voxel=36))
    assert voxels and all(v.ray_count == 40 for v in voxels)


def test_slam_archive_uses_provider_rounded_horizontal_ray_origin():
    origin = surveyor_frontend.SlamScanArchive._sensor_origins[0]

    assert origin[0].view(np.uint32) == 0xA4F20000
    np.testing.assert_array_equal(
        origin,
        np.asarray(
            [-1.0495077029659683e-16, 0.076, 0.0246], dtype=np.float32
        ),
    )


def test_discarded_batch_inference_tolerates_collator_phase_offset():
    retained = np.asarray(
        [1_000_000_000, 1_050_000_000, 1_151_000_000, 1_302_000_000],
        dtype=np.int64,
    )
    accumulated = np.asarray(
        [
            1_006_000_000,
            1_056_000_000,
            1_070_000_000,
            1_121_000_000,
            1_151_000_000,
            1_201_000_000,
            1_251_000_000,
            1_301_000_000,
        ],
        dtype=np.int64,
    )

    discarded = surveyor_frontend.infer_discarded_batch_timestamps_ns(
        retained, accumulated
    )

    assert discarded == (1_121_000_000, 1_201_000_000, 1_251_000_000)


def test_split_surfel_merge_uses_retained_empty_secondary_normal():
    previous = surveyor_frontend.SplitSurfelStatistics(
        keys=np.asarray([[6, 48, -5], [6, 49, -5]], dtype=np.int64),
        weights=np.asarray([131, 889], dtype=np.float32),
        counts=np.asarray([131, 889], dtype=np.uint32),
        means=np.asarray(
            [
                [0.5946552157, 4.8450803757, -0.4733890891],
                [0.5995074511, 4.8576812744, -0.5055425763],
            ],
            dtype=np.float32,
        ),
        covariances=np.asarray(
            [
                [
                    [0.0007773680, -0.0000364321, -0.0002173214],
                    [-0.0000364316, 0.0000415223, 0.0000344997],
                    [-0.0002173214, 0.0000344996, 0.0006043968],
                ],
                [
                    [0.0008170299, -0.0000038408, 0.0000030662],
                    [-0.0000038406, 0.0001000938, -0.0000776775],
                    [0.0000030662, -0.0000776777, 0.0007603249],
                ],
            ],
            dtype=np.float32,
        ),
        secondary_weights=np.asarray([0, 534], dtype=np.float32),
        secondary_counts=np.asarray([0, 534], dtype=np.uint32),
        secondary_means=np.asarray(
            [
                [0.5982933044, 4.8476266861, -0.4846128225],
                [0.6023314595, 4.8571314812, -0.5102235675],
            ],
            dtype=np.float32,
        ),
        secondary_covariances=np.asarray(
            [
                [
                    [0.0005516891, 0.0000026216, -0.0000410396],
                    [0.0000026226, 0.0000033086, -0.0000173295],
                    [-0.0000410397, -0.0000173281, 0.0004943897],
                ],
                [
                    [0.0007391446, -0.0000040863, 0.0000093179],
                    [-0.0000040864, 0.0000330169, -0.0000243177],
                    [0.0000093180, -0.0000243175, 0.0005907689],
                ],
            ],
            dtype=np.float32,
        ),
        is_split=np.asarray([0, 1], dtype=np.uint8),
        split_normals=np.asarray(
            [
                [0.7084200382, -0.3602244258, 0.6069427133],
                [0.7084200382, -0.3602244258, 0.6069427133],
            ],
            dtype=np.float32,
        ),
        viewpoints=np.asarray(
            [
                [0.1547417343, 3.3118679523, -0.0471562594],
                [0.1144167334, 3.0332233906, -0.0700918138],
            ],
            dtype=np.float32,
        ),
        secondary_viewpoints=np.asarray(
            [
                [-1.8519761562, 0.9478430748, -0.3090775907],
                [-1.0523788929, 0.5108903646, -0.3298420310],
            ],
            dtype=np.float32,
        ),
        primary_dirty=np.zeros(2, dtype=np.uint8),
        secondary_dirty=np.zeros(2, dtype=np.uint8),
    )
    points = np.asarray(
        [
            [0.6027607918, 4.8492059708, -0.5191903114],
            [0.6052106619, 4.8415226936, -0.5275059938],
            [0.5643694997, 4.8673405647, -0.5432150364],
        ],
        dtype=np.float32,
    )
    origins = np.broadcast_to(
        np.asarray([0.8128767014, 3.2438046932, -0.0069378614], np.float32),
        points.shape,
    ).copy()

    result = surveyor_frontend.update_split_surfel_statistics(
        previous, points, origins, 0.1, 0.05
    )

    np.testing.assert_array_equal(result.counts, [0, 1023])
    np.testing.assert_array_equal(result.secondary_counts, [0, 534])
    np.testing.assert_array_equal(
        result.means[1],
        np.asarray([0.5988605022, 4.8560528755, -0.5014968514], np.float32),
    )


def test_deferred_overlap_surfel_activation_matches_one_shot_state():
    rng = np.random.default_rng(17)
    points = rng.normal(size=(3000, 3)).astype(np.float32)
    origins = (0.2 * rng.normal(size=(3000, 3))).astype(np.float32)
    expected = surveyor_frontend.update_split_surfel_statistics(
        None, points, origins, 0.1, 0.05
    )
    actual = None
    for point_batch, origin_batch in zip(
        np.array_split(points, 7), np.array_split(origins, 7)
    ):
        actual = surveyor_frontend.update_split_surfel_statistics(
            actual,
            point_batch,
            origin_batch,
            0.1,
            0.05,
            maintain_surfels=False,
        )
    assert actual is not None
    surveyor_frontend.maintain_deferred_split_surfel_statistics(
        actual, 0.1, 0.05
    )
    for field in (
        "keys",
        "weights",
        "counts",
        "means",
        "covariances",
        "secondary_weights",
        "secondary_counts",
        "secondary_means",
        "secondary_covariances",
        "is_split",
        "split_normals",
        "viewpoints",
        "secondary_viewpoints",
        "primary_dirty",
        "secondary_dirty",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_adaptive_first_point_filter_uses_binary_nonrandom_indices():
    points = np.column_stack(
        (np.arange(8, dtype=np.float64), np.zeros(8), np.zeros(8))
    )
    normals = np.column_stack(
        (np.zeros(8), np.ones(8), np.zeros(8))
    )
    filtered_points, filtered_normals = adaptive_first_point_filter(
        points,
        normals,
        minimum_voxel_m=0.02,
        maximum_voxel_m=0.40,
        maximum_points=5,
        maximum_iterations=10,
    )
    # floor(k * 8 / 5) + 1 -> 1, 2, 4, 5, 7.
    expected = np.asarray([1, 2, 4, 5, 7])
    np.testing.assert_array_equal(filtered_points, points[expected])
    np.testing.assert_array_equal(filtered_normals, normals[expected])


def test_range_measurement_centroid_filter_preserves_octant_first_seen_order():
    points = np.asarray(
        [
            [0.01, 0.01, 0.01],
            [-0.01, -0.01, -0.01],
            [0.03, 0.01, 0.01],
            [-0.03, -0.01, -0.01],
            [0.09, 0.01, 0.01],
        ],
        dtype=np.float32,
    )
    origins = np.arange(15, dtype=np.float32).reshape((-1, 3))
    filtered_points, filtered_origins = range_measurement_centroid_filter(
        points, origins, 0.04
    )

    # Negative octant 0 precedes positive octant 7. Within an octant, voxel
    # output follows first occurrence in the input. The two positive samples
    # in voxel (0, 0, 0) use the captured float running-centroid formula.
    expected_points = np.asarray(
        [[-0.02, -0.01, -0.01], [0.02, 0.01, 0.01], [0.09, 0.01, 0.01]],
        dtype=np.float32,
    )
    expected_origins = np.asarray(
        [[6.0, 7.0, 8.0], [3.0, 4.0, 5.0], [12.0, 13.0, 14.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(filtered_points, expected_points)
    np.testing.assert_array_equal(filtered_origins, expected_origins)


def test_slam_map_odom_fusion_and_five_times_upsampling():
    timestamps = np.array([0.0, 1.0, 2.0])
    local = Trajectory(
        timestamps,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
    )
    map_from_odom = Pose(
        0.0,
        np.array([3.0, -2.0, 1.0]),
        np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
    )
    global_slam = Trajectory(
        timestamps,
        np.asarray([map_from_odom.apply(point) for point in local.translations]),
        np.tile(map_from_odom.quaternion_xyzw, (3, 1)),
    )
    fused = fuse_global_and_local(global_slam, local, upsampling_factor=5)
    assert len(fused.timestamps) == 11
    np.testing.assert_allclose(fused.translations[::5], global_slam.translations, atol=1e-12)
    evaluation = evaluate_trajectory(fused, fused)
    assert evaluation["position_error_m"]["max"] < 1e-12
    assert evaluation["rotation_error_deg"]["max"] < 1e-12


def test_slam_relative_error_is_global_gauge_invariant():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    reference = Trajectory(
        timestamps,
        np.column_stack((timestamps, np.zeros((4, 2)))),
        np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
    )
    map_from_reference = Pose(
        0.0,
        np.array([4.0, -3.0, 2.0]),
        np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
    )
    estimate = Trajectory(
        timestamps,
        np.asarray([map_from_reference.apply(point) for point in reference.translations]),
        np.tile(map_from_reference.quaternion_xyzw, (4, 1)),
    )
    relative = evaluate_relative_trajectory(estimate, reference, delta_seconds=1.0)
    assert relative["translation_error_m"]["max"] < 1e-12
    assert relative["rotation_error_deg"]["max"] < 1e-12


def test_surveyor_pose_graph_uses_submap_memberships_and_loops():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    gravity = np.array([0.0, 0.0, 9.81])
    nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            index * 50_000_000,
            Rigid3(np.array([float(index), 0.0, 0.0]), identity),
            Rigid3(np.array([float(index), 0.0, 0.0]), identity),
            gravity,
        )
        for index in range(3)
    )
    submaps = (
        Submap(
            NodeId(0, 0), 0, 100_000_000,
            Rigid3(np.array([0.0, 0.0, 0.0]), identity),
            (0, 1), True, gravity,
        ),
        Submap(
            NodeId(0, 1), 50_000_000, 150_000_000,
            Rigid3(np.array([1.0, 0.0, 0.0]), identity),
            (1, 2), True, gravity,
        ),
    )
    loops = (
        LoopConstraint(
            NodeId(0, 0), NodeId(0, 2),
            Rigid3(np.array([2.2, 0.0, 0.0]), identity),
            10.0, 10.0, True, 1,
        ),
    )
    problem = build_pose_graph(
        nodes, submaps, loops,
        odometry_translation_weight=10.0,
        odometry_rotation_weight=10.0,
    )
    assert len(problem.edges) == 5  # four memberships plus one loop
    assert sum(edge.kind == "odometry" for edge in problem.edges) == 4
    result = optimize_pose_graph(
        problem, max_iterations=50, robust_loss="linear"
    )
    assert result.success
    assert result.final_cost < result.initial_cost * 0.5
    # Gauge fixing the first submap must survive optimization exactly.
    fixed = problem.fixed_vertices[0]
    np.testing.assert_allclose(
        result.poses[fixed].translation,
        problem.initial_poses[fixed].translation,
        atol=1e-12,
    )


def test_imu_midpoint_preintegration_preserves_stationary_motion_model():
    gravity = 9.81
    samples = tuple(
        ImuSample(
            timestamp,
            np.array([0.0, 0.0, gravity]),
            np.zeros(3),
        )
        for timestamp in (0, 10_000_000, 20_000_000)
    )
    edge = preintegrate_imu(
        samples, 0, 20_000_000, ImuCalibration(gravity_magnitude=gravity)
    )
    np.testing.assert_allclose(edge.delta_rotation_xyzw, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(edge.delta_velocity, [0.0, 0.0, gravity * 0.02])
    np.testing.assert_allclose(
        edge.delta_position, [0.0, 0.0, 0.5 * gravity * 0.02**2]
    )


def test_raw_imu_tracker_integrates_constant_yaw_at_lidar_times():
    samples = tuple(
        ImuSample(
            timestamp,
            np.array([0.0, 0.0, 9.81]),
            np.array([0.0, 0.0, 0.1]),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        for timestamp in (0, 1_000_000_000, 2_000_000_000)
    )
    tracker = RawImuTracker(samples)
    actual = Rotation.from_quat(
        tracker.orientations_at([0, 500_000_000, 1_000_000_000])
    )
    expected = Rotation.from_rotvec(
        np.array([[0.0, 0.0, angle] for angle in (0.0, 0.05, 0.1)])
    )
    errors = (actual.inv() * expected).magnitude()
    np.testing.assert_allclose(errors, 0.0, atol=1.0e-12)


def test_raw_imu_tracker_preserves_firmware_quaternion_until_first_update():
    raw_orientation = np.array([0.1, -0.2, 0.3, 0.9]) * 1.00000004
    samples = tuple(
        ImuSample(
            timestamp,
            np.array([0.0, 0.0, 9.81]),
            np.zeros(3),
            orientation_xyzw=raw_orientation,
        )
        for timestamp in (0, 1_000_000_000)
    )

    np.testing.assert_array_equal(samples[0].orientation_xyzw, raw_orientation)
    tracker = RawImuTracker(samples)
    np.testing.assert_array_equal(tracker.orientations_at([0])[0], raw_orientation)


def test_rigid3_preserves_non_unit_coefficients_but_exposes_a_rotation():
    coefficients = np.array([0.1, -0.2, 0.3, 0.9]) * 1.00000004
    pose = Rigid3(np.zeros(3), coefficients)

    np.testing.assert_array_equal(pose.quaternion_xyzw, coefficients)
    np.testing.assert_allclose(np.linalg.norm(pose.rotation.as_quat()), 1.0)


def test_raw_pose_predictor_uses_corrected_constant_velocity_for_deskew():
    samples = tuple(
        ImuSample(
            timestamp,
            np.array([0.0, 0.0, 9.81]),
            np.zeros(3),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        for timestamp in (0, 1_000_000_000, 2_000_000_000)
    )
    predictor = RawConstantVelocityPosePredictor(samples)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    predictor.correct(0, Rigid3(np.zeros(3), identity))
    predictor.correct(
        1_000_000_000, Rigid3(np.array([1.0, 0.0, 0.0]), identity)
    )
    rotations, translations = predictor.relative_motion(
        [1_250_000_000, 1_500_000_000], 1_500_000_000
    )
    np.testing.assert_allclose(
        Rotation.from_quat(rotations).magnitude(), 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        translations,
        np.array([[-0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1.0e-12,
    )


def test_surveyor_imu_pose_graph_preserves_stationary_solution():
    gravity = 9.81
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    gravity_observation = np.array([0.0, 0.0, gravity])
    nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            index * 10_000_000,
            Rigid3(np.zeros(3), identity),
            Rigid3(np.zeros(3), identity),
            gravity_observation,
        )
        for index in range(3)
    )
    submaps = (
        Submap(
            NodeId(0, 0),
            0,
            20_000_000,
            Rigid3(np.zeros(3), identity),
            (0, 1, 2),
            True,
            gravity_observation,
        ),
    )
    samples = tuple(
        ImuSample(timestamp, gravity_observation, np.zeros(3))
        for timestamp in (0, 10_000_000, 20_000_000)
    )
    problem = build_imu_pose_graph(
        build_pose_graph(nodes, submaps, ()),
        nodes,
        samples,
        ImuCalibration(gravity_magnitude=gravity),
    )
    assert problem.samples == samples
    result = optimize_imu_pose_graph(
        problem, max_iterations=5, robust_loss="linear"
    )
    assert result.success
    assert result.initial_cost < 1.0e-20
    assert result.final_cost < 1.0e-20
    for pose in result.poses:
        np.testing.assert_allclose(pose.translation, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(pose.quaternion_xyzw, identity, atol=1.0e-12)


def test_surveyor_imu_initial_velocity_uses_integer_nanosecond_deltas():
    base_timestamp_ns = 1_784_626_878_213_174_336
    timestamps_ns = np.array(
        [
            base_timestamp_ns,
            base_timestamp_ns + 50_128_715,
            base_timestamp_ns + 101_719_548,
        ],
        dtype=np.int64,
    )
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    observation = np.array([0.0, 0.0, 9.81])
    positions = np.array(
        [
            [-0.2934807430954472, -0.19610578753327734, 0.9656537961245656],
            [-0.29449935186006376, -0.19554688326476397, 0.9723123625926003],
            [-0.29579806530566594, -0.1950081696185888, 0.9782832500332568],
        ]
    )
    nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            int(timestamp_ns),
            Rigid3(position, identity),
            Rigid3(position, identity),
            observation,
        )
        for index, (timestamp_ns, position) in enumerate(
            zip(timestamps_ns, positions)
        )
    )
    submaps = (
        Submap(
            NodeId(0, 0),
            int(timestamps_ns[0]),
            int(timestamps_ns[-1]),
            Rigid3(np.zeros(3), identity),
            (0, 1, 2),
            True,
            observation,
        ),
    )
    samples = tuple(
        ImuSample(int(timestamp_ns), observation, np.zeros(3))
        for timestamp_ns in timestamps_ns
    )

    problem = build_imu_pose_graph(
        build_pose_graph(nodes, submaps, ()),
        nodes,
        samples,
        ImuCalibration(gravity_magnitude=9.81),
    )

    durations_seconds = np.diff(timestamps_ns).astype(np.float64) / 1.0e9
    segment_velocities = np.diff(positions, axis=0) / durations_seconds[:, None]
    expected = np.vstack((segment_velocities[0], segment_velocities))
    np.testing.assert_array_equal(problem.initial_velocities, expected)


def test_fast_imu_pose_graph_preserves_stationary_solution():
    gravity = ImuCalibrationOptions().gravity_magnitude
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    observation = np.array([0.0, 0.0, gravity])
    nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            index * 10_000_000,
            Rigid3(np.zeros(3), identity),
            Rigid3(np.zeros(3), identity),
            observation,
        )
        for index in range(3)
    )
    submaps = (
        Submap(
            NodeId(0, 0),
            0,
            20_000_000,
            Rigid3(np.zeros(3), identity),
            (0, 1, 2),
            True,
            observation,
        ),
    )
    samples = tuple(
        ImuSample(timestamp, observation, np.zeros(3))
        for timestamp in (0, 10_000_000, 20_000_000)
    )
    problem = build_fast_imu_pose_graph(
        build_pose_graph(nodes, submaps, ()),
        nodes,
        samples,
        ImuCalibration(gravity_magnitude=gravity),
    )
    assert len(problem.acceleration_factors) == 1
    assert len(problem.rotation_factors) == 2
    result = optimize_fast_imu_pose_graph(problem, max_iterations=5)
    assert result.success
    assert result.initial_cost < 1.0e-20
    assert result.final_cost < 1.0e-20
    for pose in result.poses:
        np.testing.assert_allclose(pose.translation, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(pose.quaternion_xyzw, identity, atol=1.0e-12)


def test_fast_imu_acceleration_loss_uses_integer_midpoint_interval():
    gravity = ImuCalibrationOptions().gravity_magnitude
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    observation = np.array([0.0, 0.0, gravity])
    timestamps = (0, 10_000_001, 20_000_003)
    nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            timestamp,
            Rigid3(np.zeros(3), identity),
            Rigid3(np.zeros(3), identity),
            observation,
        )
        for index, timestamp in enumerate(timestamps)
    )
    submaps = (
        Submap(
            NodeId(0, 0),
            timestamps[0],
            timestamps[-1],
            Rigid3(np.zeros(3), identity),
            (0, 1, 2),
            True,
            observation,
        ),
    )
    samples = tuple(
        ImuSample(timestamp, observation, np.zeros(3)) for timestamp in timestamps
    )
    problem = build_fast_imu_pose_graph(
        build_pose_graph(nodes, submaps, ()),
        nodes,
        samples,
        ImuCalibration(gravity_magnitude=gravity),
    )
    factor = problem.acceleration_factors[0]
    assert factor.loss_duration == 0.010000002
    assert factor.loss_duration != 0.5 * (
        factor.first_duration + factor.second_duration
    )


def test_fast_imu_clips_after_rotating_raw_acceleration_endpoints():
    sample_times = (0, 1_000_000_000, 2_000_000_000)
    accelerations = (
        np.array([1.0, 0.2, 9.0]),
        np.array([0.5, -0.3, 9.5]),
        np.array([-0.2, 0.4, 10.0]),
    )
    angular_velocities = (
        np.array([0.0, 0.0, 0.1]),
        np.array([0.0, 0.0, 0.3]),
        np.array([0.0, 0.0, 0.5]),
    )

    def clipped(first, second, left_clip, right_clip):
        left = left_clip * second + (1.0 - left_clip) * first
        right = right_clip * first + (1.0 - right_clip) * second
        return (left + right) * 0.5 * (1.0 - left_clip - right_clip)

    first_delta_angle = clipped(
        angular_velocities[0], angular_velocities[1], 0.5, 0.0
    )
    first_rotation = Rotation.from_rotvec(first_delta_angle)
    first_velocity = clipped(
        accelerations[0], first_rotation.apply(accelerations[1]), 0.5, 0.0
    )
    second_delta_angle = clipped(
        angular_velocities[1], angular_velocities[2], 0.0, 0.5
    )
    final_rotation = first_rotation * Rotation.from_rotvec(second_delta_angle)
    second_velocity = clipped(
        first_rotation.apply(accelerations[1]),
        final_rotation.apply(accelerations[2]),
        0.0,
        0.5,
    )

    rotation, velocity = _integrate_fast_imu(
        sample_times,
        tuple(zip(accelerations, angular_velocities)),
        500_000_000,
        1_500_000_000,
    )
    np.testing.assert_allclose(rotation.as_quat(), final_rotation.as_quat(), atol=1e-15)
    np.testing.assert_allclose(velocity, first_velocity + second_velocity, atol=1e-15)


def test_live_imu_calibration_uses_installed_stage2_priors():
    options = ImuCalibrationOptions()
    assert options.gravity_magnitude == 9.807232
    assert options.gravity_prior_weight == 1.0e4
    assert options.imu_orientation_prior_weight == 5.0e4
    assert options.linear_acceleration_bias_prior_weight == (1.0e5, 1.0e5, 1.0e2)
    assert options.linear_acceleration_scaling_prior_weight == (1.0e5,) * 3
    assert options.angular_velocity_bias_prior_weight == (0.0,) * 3
    assert options.angular_velocity_scaling_prior_weight == (1.0e4,) * 3

    gravity = 9.81
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    observation = np.array([0.0, 0.0, gravity])
    nodes = tuple(
        TrajectoryNode(
            NodeId(0, index),
            index * 10_000_000,
            Rigid3(np.zeros(3), identity),
            Rigid3(np.zeros(3), identity),
            observation,
        )
        for index in range(3)
    )
    submaps = (
        Submap(
            NodeId(0, 0),
            0,
            20_000_000,
            Rigid3(np.zeros(3), identity),
            (0, 1, 2),
            True,
            observation,
        ),
    )
    samples = tuple(
        ImuSample(timestamp, observation, np.zeros(3))
        for timestamp in (0, 10_000_000, 20_000_000)
    )
    problem = build_imu_pose_graph(
        build_pose_graph(nodes, submaps, ()),
        nodes,
        samples,
        ImuCalibration(gravity_magnitude=gravity),
    )
    result = optimize_imu_pose_graph(
        problem,
        max_iterations=2,
        robust_loss="linear",
        calibrate_imu_intrinsics=True,
    )
    assert result.calibration is not None
    assert result.final_cost < result.initial_cost


def test_surveyor_point_to_plane_frontend_recovers_rigid_motion():
    rng = np.random.default_rng(4)
    source = rng.uniform(-2.0, 2.0, (1000, 3))
    expected = Rigid3(
        np.array([0.05, -0.04, 0.03]),
        Rotation.from_euler("xyz", [1.0, -1.5, 2.0], degrees=True).as_quat(),
    )
    target = expected.rotation.apply(source) + expected.translation
    normals = rng.normal(size=source.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    result = point_to_plane_icp(
        source,
        target,
        normals,
        max_correspondence_m=0.3,
        min_correspondences=100,
        max_iterations=20,
    )
    assert result.converged and result.overlap == 1.0
    np.testing.assert_allclose(
        result.target_from_source.translation, expected.translation, atol=1e-10
    )
    rotation_error = (
        result.target_from_source.rotation.inv() * expected.rotation
    ).magnitude()
    assert rotation_error < 1.0e-10


def test_binary_point_to_plane_frontend_returns_final_information_matrix():
    rng = np.random.default_rng(41)
    points = rng.uniform(-2.0, 2.0, (1000, 3)).astype(np.float32)
    normals = rng.normal(size=points.shape).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    result = point_to_plane_icp(
        points,
        points,
        normals,
        binary_compatible=True,
        max_correspondence_m=0.3,
        huber_m=float("inf"),
        max_iterations=6,
        min_iterations=6,
        correspondence_levels_m=(0.3,),
        initial_plane_distance_m=0.2,
        contracted_plane_distance_m=0.03,
        contraction_iterations=6,
        min_correspondences=100,
        compute_information_matrix=True,
    )
    assert result.information_matrix is not None
    assert result.information_matrix.shape == (6, 6)
    np.testing.assert_array_equal(
        result.information_matrix, result.information_matrix.T
    )
    assert np.all(np.linalg.eigvalsh(result.information_matrix) > 0.0)


def test_binary_point_to_plane_uses_float_contraction_boundary():
    maximum_limit = np.float32(0.2)
    minimum_limit = np.float32(0.02)
    contraction = np.float32(np.float32(4.0) / np.float32(5.0))
    boundary = np.float32(
        np.float32(contraction * np.float32(minimum_limit - maximum_limit))
        + maximum_limit
    )
    assert boundary.view(np.uint32) == 0x3D656040

    source = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    target = np.asarray(
        [[np.float32(1.0) - boundary, 0.0, 0.0]], dtype=np.float32
    )
    normals = np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32)
    identity = Rigid3(
        np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0])
    )
    correspondence_counts = []

    def identity_step(points, _targets, _normals, _normalization):
        correspondence_counts.append(len(points))
        return identity, np.zeros(6), 1.0

    with patch.object(
        surveyor_frontend,
        "_binary_point_plane_step",
        side_effect=identity_step,
    ):
        result = point_to_plane_icp(
            source,
            (target,),
            (normals,),
            identity,
            source_origins=np.zeros_like(source),
            binary_compatible=True,
            max_correspondence_m=0.15,
            huber_m=float("inf"),
            max_iterations=6,
            min_iterations=6,
            correspondence_levels_m=(0.15,),
            initial_plane_distance_m=0.2,
            contracted_plane_distance_m=0.02,
            contraction_iterations=6,
            min_correspondences=1,
            max_incidence_angle_deg=86.0,
            num_threads=1,
        )

    # The residual is exactly equal to the fourth interpolated float limit.
    # The installed strict comparison rejects equality before solve five.
    assert correspondence_counts == [1, 1, 1, 1]
    assert result.iterations == 5


def test_binary_point_to_plane_divides_incidence_after_dot_product():
    source = np.asarray(
        [[2.1450124, 4.3833275, -0.4696229]], dtype=np.float32
    )
    origins = np.asarray(
        [[2.13105, -3.6259105, -1.3118596]], dtype=np.float32
    )
    target = np.asarray(
        [[2.1340845, 4.396613, -0.39986402]], dtype=np.float32
    )
    normals = np.asarray(
        [[0.982472, -0.08907371, 0.16375132]], dtype=np.float32
    )
    identity = Rigid3(
        np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0])
    )
    correspondence_counts = []

    def identity_step(points, _targets, _normals, _normalization):
        correspondence_counts.append(len(points))
        return identity, np.zeros(6), 1.0

    with patch.object(
        surveyor_frontend,
        "_binary_point_plane_step",
        side_effect=identity_step,
    ):
        result = point_to_plane_icp(
            source,
            (target,),
            (normals,),
            identity,
            source_origins=origins,
            binary_compatible=True,
            max_correspondence_m=0.15,
            huber_m=float("inf"),
            max_iterations=1,
            min_iterations=1,
            correspondence_levels_m=(0.15,),
            initial_plane_distance_m=0.2,
            contracted_plane_distance_m=0.02,
            contraction_iterations=6,
            min_correspondences=1,
            max_incidence_angle_deg=86.0,
            num_threads=1,
        )

    # A component-wise ray normalization rounds the cosine one ULP below the
    # 86-degree limit.  The installed dot-then-divide grouping reaches the
    # limit exactly, and its strict comparison keeps equality.
    assert correspondence_counts == [1]
    assert result.converged


def test_surveyor_submaps_keep_two_active_distance_windows():
    config = FrontendConfig(
        submap_overlap_displacement_m=5.0,
        submap_overlap_path_m=5.0,
        submap_finish_displacement_m=10.0,
        submap_finish_path_m=10.0,
        map_rebuild_interval=100,
    )
    builder = OverlappingSubmapBuilder(config)
    points = np.array(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    for index in range(15):
        builder.add(
            FrontendNode(
                NodeId(0, index),
                index * 50_000_000,
                Rigid3(
                    np.array([float(index), 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0, 1.0]),
                ),
                points,
                normals,
                None,
            )
        )
    builder.finish()
    # Threshold node 10 anchors the third submap but is not inserted into it;
    # its 11--14 support remains below 5 m at EOF, so it is discarded exactly
    # like the binary's final half-built submap.
    assert [submap.node_indices[0] for submap in builder.submaps] == [0, 6]
    assert builder.submaps[0].node_indices == list(range(0, 11))
    assert builder.submaps[1].node_indices == list(range(6, 15))


def test_scan_context_loop_descriptor_is_yaw_invariant():
    rng = np.random.default_rng(8)
    cloud = rng.normal(size=(2000, 3)) * np.array([4.0, 2.0, 1.0])
    rotated = Rotation.from_euler("z", 36.0, degrees=True).apply(cloud)
    first = scan_context_descriptor(cloud)
    second = scan_context_descriptor(rotated)
    distance, shift = descriptor_distance(first, second)
    assert distance < 1.0e-10
    assert shift in (6, 54)


def test_fast_correlative_loop_search_recovers_grid_hypothesis():
    rng = np.random.default_rng(18)
    source = rng.uniform(-3.0, 3.0, (600, 3))
    expected = Rigid3(
        np.array([0.2, -0.2, 0.0]),
        Rotation.from_euler("z", 3.0, degrees=True).as_quat(),
    )
    target = expected.rotation.apply(source) + expected.translation
    result = fast_correlative_scan_match(
        source,
        target,
        Rigid3(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
        FrontendConfig(),
    )
    assert result.score > 0.999
    np.testing.assert_allclose(
        result.target_from_source.translation, expected.translation, atol=1e-12
    )
    assert (
        result.target_from_source.rotation.inv() * expected.rotation
    ).magnitude() < 1.0e-12


def test_hybrid_probability_grid_uses_quantized_cartographer_score():
    grid = HybridProbabilityGrid(
        0.2,
        indices=np.array([[5, 0, 0], [6, 0, 0]], dtype=np.int32),
        values=np.array([32767, 1], dtype=np.uint16),
    )
    try:
        assert grid.cell_count == 2
        score = grid.score(np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))
        np.testing.assert_allclose(score, 0.5, atol=1.0e-7)

        live = HybridProbabilityGrid(0.2)
        try:
            live.insert(
                np.array([[1.0, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0]]),
            )
            np.testing.assert_allclose(
                live.score(np.array([[1.0, 0.0, 0.0]])), 0.9, atol=1.0e-7
            )
        finally:
            live.close()
    finally:
        grid.close()


def test_loop_constraint_threshold_boundaries_match_binary_semantics():
    config = FrontendConfig()
    assert rotational_score_is_acceptable(0.77, config)
    assert not rotational_score_is_acceptable(
        np.nextafter(0.77, -np.inf), config
    )

    strength_boundary = np.diag(
        [50.0, 75.0, 100.0, 125.0, 150.0, np.nextafter(2500.0, 0.0)]
    )
    accepted = evaluate_constraint_stability(strength_boundary, config)
    assert accepted.is_stable
    assert accepted.strength == 50.0
    assert accepted.anisotropy < 50.0

    weak = np.diag(
        [np.nextafter(50.0, 0.0), 75.0, 100.0, 125.0, 150.0, 200.0]
    )
    assert not evaluate_constraint_stability(weak, config).is_stable

    anisotropy_boundary = np.diag(
        [50.0, 75.0, 100.0, 125.0, 150.0, 2500.0]
    )
    rejected = evaluate_constraint_stability(anisotropy_boundary, config)
    assert rejected.anisotropy == 50.0
    assert not rejected.is_stable


def test_detect_loop_constraints_applies_rotational_and_stability_gates():
    identity = Rigid3(
        np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    )
    points = np.array(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    nodes = (
        FrontendNode(NodeId(0, 0), 0, identity, points, normals, None),
        FrontendNode(
            NodeId(0, 100),
            500_000_000,
            Rigid3(
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            points,
            normals,
            None,
        ),
    )
    submap = surveyor_frontend.FrontendSubmap(
        NodeId(0, 0), identity, 0, 0.0, [0], 0, True
    )
    submap._cached_levels = [(points, normals) for _ in range(3)]
    submap._cached_points, submap._cached_normals = submap._cached_levels[0]

    stable_matrix = np.diag(
        [50.0, 75.0, 100.0, 125.0, 150.0, np.nextafter(2500.0, 0.0)]
    )

    def run(rotational_score, information_matrix):
        correlative = CorrelativeResult(identity, 1.0, 0.0, 1, rotational_score)
        icp = IcpResult(
            identity,
            0.0,
            1.0,
            len(points),
            6,
            False,
            0.0,
            information_matrix,
        )
        with patch.object(
            surveyor_frontend,
            "fast_correlative_scan_match",
            return_value=correlative,
        ), patch.object(
            surveyor_frontend, "point_to_plane_icp", return_value=icp
        ):
            return detect_loop_constraints(nodes, (submap,))

    try:
        assert len(run(0.77, stable_matrix)) == 1
        assert not run(np.nextafter(0.77, -np.inf), stable_matrix)
        assert not run(0.77, np.diag([50.0] * 5 + [2500.0]))
        assert not run(
            0.77,
            np.diag([np.nextafter(50.0, 0.0)] + [100.0] * 5),
        )

        immediate_node = FrontendNode(
            NodeId(0, 10),
            500_000_000,
            nodes[1].local_pose,
            points,
            normals,
            None,
        )
        low_rotational_score = CorrelativeResult(
            identity, 0.0, 0.0, 1, np.nextafter(0.77, -np.inf)
        )
        stable_icp = IcpResult(
            identity, 0.0, 1.0, len(points), 6, False, 0.0, stable_matrix
        )
        with patch.object(
            surveyor_frontend,
            "fast_correlative_scan_match",
            return_value=low_rotational_score,
        ), patch.object(
            surveyor_frontend, "point_to_plane_icp", return_value=stable_icp
        ):
            assert len(detect_loop_constraints((nodes[0], immediate_node), (submap,))) == 1
    finally:
        submap.hybrid_grid.close()


def test_submap_gravity_accumulator_preserves_binary_operation_order():
    submap_pose = Rigid3(
        np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    )
    node_poses = (
        Rigid3(
            np.zeros(3),
            np.array(
                [
                    0.09682004706828,
                    0.037601158556655065,
                    -0.7076787610721457,
                    0.6988581525869458,
                ]
            ),
        ),
        Rigid3(
            np.zeros(3),
            np.array(
                [
                    0.09732353457919801,
                    0.03750855066302696,
                    -0.7087189965113236,
                    0.6977382189819855,
                ]
            ),
        ),
    )
    observations = (
        np.array([-1.8598854201242871, 0.8054783987945041, 9.598340000910811]),
        np.array([-1.819068330679075, 0.787932188470071, 9.607614338361694]),
    )
    state = np.zeros(3)
    for count, (pose, observation) in enumerate(zip(node_poses, observations)):
        state = surveyor_frontend._updated_submap_gravity(
            state, count, submap_pose, pose, observation
        )
    assert state.view(np.uint64).tolist() == [
        13801456347557965624,
        13806001469030890104,
        4621712136161014726,
    ]


def test_submap_rotation_preserves_gravity_alignment_cancellation_bits():
    node_pose = Rigid3(
        np.zeros(3),
        np.array(
            [
                0.07484415932340763,
                0.032153971847607654,
                -0.6836425955927617,
                0.7252566962123047,
            ]
        ),
    )
    gravity = np.array(
        [0.253571876005113, -0.08758642456192368, 9.806331114230833]
    )
    quaternion = surveyor_frontend._submap_rotation_from_node(
        node_pose, gravity
    )
    assert quaternion.view(np.uint64).tolist() == [
        4591123337114089256,
        13808088477235436544,
        4363988038922010624,
        4607141642013218076,
    ]


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
