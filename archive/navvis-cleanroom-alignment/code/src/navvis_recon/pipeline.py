"""Readable reconstruction of navvis-postprocessing's action graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    order: int
    name: str
    implementation: str
    role: str


# Names/order are recovered from frozen libppp action modules and the reference
# postprocessing log. Frontend stages now emit the same node/submap/constraint
# vocabulary as the recovered sparse IMU graph, while their numerical
# thresholds remain independently measurable against frozen binary artifacts.
ACTIONS = (
    Action(10, "setup/license/integrity/reindex/rewrite", "external", "validate and normalize recording"),
    Action(40, "SLAM input filtering", "C++ Pandar decoder + surveyor_frontend.filter_scan", "calibrated range decode, fringe/body filters and scan voxels"),
    Action(41, "local trajectory builder", "surveyor_frontend.SurveyorFrontend.process", "motion prediction and robust point-to-plane scan-to-submap matching"),
    Action(42, "submap construction", "surveyor_frontend.OverlappingSubmapBuilder", "overlapping 0.2 m surfel submaps and short-tail rejection"),
    Action(43, "loop-closure constraints", "surveyor_frontend.detect_loop_constraints", "overlap candidates, fast-correlative search and geometric ICP acceptance"),
    Action(44, "sparse pose graph Stage 1", "native Ceres + surveyor_slam.build_fast_imu_pose_graph", "pose-only graph with exact fast-IMU factors and loop loss"),
    Action(45, "sparse pose graph Stage 2", "surveyor_slam.optimize_imu_pose_graph", "joint pose, velocity, gravity, IMU orientation, bias and scale solve"),
    Action(46, "trajectory upsampling", "slam_reconstruction.fuse_global_and_local", "map/odom fusion and 5x support"),
    Action(50, "capture locations", "external", "associate capture timestamps with trajectory"),
    Action(55, "Wi-Fi export", "quality_map.export_radio_observations", "normalize passive radio observations"),
    Action(60, "mapped-space quality", "quality_map.mapped_space_quality", "count rays through 1/6 m voxels"),
    Action(70, "Bluetooth export", "quality_map.export_radio_observations", "normalize passive radio observations"),
    Action(75, "trace", "external", "export interpolated trajectory CSV"),
    Action(76, "floor estimation", "floor_estimator.refined_floor_estimator", "segment trace by height/time"),
    Action(77, "image post-processing", "image_postprocessing.process_linear_rgb", "DNG, WB, HDR, denoise, sharpen"),
    Action(80, "cloud builder", "cloud_builder.build_cloud", "filter and unskew laser samples"),
    Action(90, "surface filter", "cloud_surface_filter.filter_surface", "voxelize, normals, freespace and outliers"),
    Action(91, "GS filter", "external", "optional post-filter action"),
    Action(96, "cleanup", "external", "remove intermediate artifacts"),
    Action(120, "cloud coloring", "pointcloud_coloring.colorize", "depth-tested multi-view color fusion"),
    Action(121, "move", "external", "place generated artifacts"),
    Action(125, "cloud conversion", "external", "write PLY/E57/tiled formats"),
    Action(130, "registration", "external", "apply supplied transforms; not trajectory SLAM"),
    Action(140, "point-cloud rendering", "panorama_rendering.render_surfels", "surfel panoramas from PLY"),
    Action(141, "operator detection", "external", "produce privacy/operator masks"),
    Action(142, "panorama rendering", "panorama_rendering.stitch_panorama", "warp, seam, blend and inpaint"),
)


def non_slam_actions() -> tuple[Action, ...]:
    return tuple(action for action in ACTIONS if not 40 <= action.order < 50)


def active_profile() -> dict:
    """Resolve the options in the supplied contest command."""
    return {
        "caller": "sitemaker",
        "device": "G11 with two PandarXTM laser models",
        "preset": "standard",
        "cloud_resolution_m": 0.01,
        "cloud_format": "ply",
        "panorama_threads": 32,
        "force": True,
        "slam": "raw scan frontend + overlapping submaps + loop ICP + two-stage sparse IMU graph; frozen Stage1 is exact while full generated Submap/loop and Stage2 remain approximate",
    }
