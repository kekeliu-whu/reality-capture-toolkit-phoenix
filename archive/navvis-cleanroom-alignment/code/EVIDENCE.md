# Reverse-engineering evidence

## Target identity

| Binary | Size | SHA-256 | GNU Build ID |
|---|---:|---|---|
| `navvis-postprocessing` | 6,019,480 | `71caf39235c8cfe18062a60cb459d99ab3b2d817f121912a122a2fcc54a29828` | `d15dd1ff734331e5bf9a34f1a7d4b404d68eb90f` |
| `cloud_builder` | 6,553,376 | `973f4596a8022e90de2af44d5aa8e16a51e033587740f4bae0d97a1bee85d6cd` | `ea5e2be5d588da812f332ef2246559db488e435c` |
| `nv_cloud-surface-filter` | 6,349,136 | `80e81bca7ca29efe812e66f172e1f7003382e1022ab671a5b46dda169b81375e` | `f763594d4f34c5dd34b0b619348b8dd685d73afa` |
| `nv_colorcloud` | 10,101,656 | `b582016681f9552cfec69471c51f3f9373a5828a2cefd3e1eddc31324545c234` | `a7586f518009434f5e97891f897aea42675f26a0` |
| `nv_image-postprocessing` | 5,611,808 | `1b9d8cf2c947f8fcd2cf3f7e981f2b5b9af4eff9acfc49d39471c634d3ebd980` | `80d6c0e0be5fe88183afc2d5a32a30a8c2c08073` |
| `nv_pointcloud-renderer` | 5,779,776 | `5cfa1e144f7ee217f2fdc37b539812840e10778d0b32af756d58f1998440c1a9` | `53290501b8fab09b521abd21e8936cd5b5236e4d` |
| `nv_panorama-renderer` | 6,578,512 | `6abadde37fa7b656212bcd2b4397504d3754e213e82bf534a9e13f0615373cdc` | `b3ce9547797e6859e8ea5ff2428eb72f20645f1b` |

Package versions are all release 6.0.7.1. Component builds are processing-pipeline b19631,
image-postprocessing b67758, panorama-rendering b67759, pointcloud-coloring b45104,
pointcloud-tools b39828 and ros-shared b54292.

The top-level executable contains Nuitka 1.3.8 runtime strings and frozen Python module
metadata. Frozen `libppp` action names establish the pipeline order recorded in
`pipeline.py`.

## Evidence by module

### Cloud builder — high confidence

- RTTI identifies processors over `PointXYZNormalITR`: `NonFiniteXYZFilter`,
  `IntensityNormalizer`, `IntensityRegionFilter`, `MultilayerFringeFilter`,
  `RegionFilter`, `RingRangeFilter` and `PlaneFilter`.
- PCL imports identify `RandomSampleConsensus<PointXYZNormalITR>` and
  `SampleConsensusModelPlane`.
- Embedded source/header names identify ordered multilayer normal computation.
- Transform-tree and interpolated-trajectory symbols, plus motion velocity/acceleration
  strings, establish per-point unskewing.
- G11-specific region constructors name horizontal XTM strut fringe and vertical XTM
  body/head reflection regions.
- Runtime RTTI identifies the vertical-foot stage as an outer
  `PlaneFilter<PointXYZNormalITR>` over a transformed `RegionFilter`. The nested boolean
  region is a 0.5 m cylinder intersected with `z<=0`; the captured transform is the
  `laser_vert` box rotation from `sensor_frame.xml` with translation intentionally unused.
- The installed PlaneFilter stores `0.3, 0.02, 0.01` at offsets `+0x78/+0x7c/+0x80`.
  Its active path constructs PCL `SampleConsensusModelPlane` plus
  `RandomSampleConsensus`, resets the deterministic model RNG to 12345, sets 50 maximum
  iterations and a 0.01 m RANSAC threshold, then rejects region points whose absolute
  distance from the recovered plane exceeds 0.02 m.

Exact G11 region coordinates and calibration tables are not present as named data, so the
reference exposes them as configuration.

For the authoritative `1770532442.4–1770532442.5` scan window, debugger-captured candidate
records and coefficients were compared against independent output. Both implementations
produce 51,480 pre-foot points, 4,238 candidates, 1,808 rejected points and 49,672 survivors;
all rejected indices agree. Survivor world-coordinate error is 1.53 micrometres mean and
0.030 millimetres maximum, and normalized intensity is bit-identical. On the complete
`2026-02-08_07.33.20` recording, the C++ decoder processed 528,916 XTM packets in 177 seconds
and wrote 24,893,273 occupied 1 cm endpoint voxels after rejecting 397,747 vertical-foot
returns.

### Cloud surface filter — high confidence structure, medium confidence thresholds

- RTTI/imports identify `AdaptiveStatisticalOutlierRemoval<PointXYZINormalWeighted>`,
  `FreespaceOctree`, `CompactOccupancyOctree` and recommended adaptive-SOR parameters.
- Recovered control flow calls, in order: global bounds, octree construction, `addPoint`,
  centroid-normal computation, ray intersections, statistics, free-voxel removal and
  compact occupancy conversion, followed by parallel tile processing.
- Messages state the free-voxel rule uses a minimum intersection count plus
  intersection/hit ratio. They also establish origin-distance bounds, shortened ray
  endpoints, incidence-angle checks and ray-to-centroid tolerance.
- Hidden CLI options establish multi-level normal radii, dynamic smoothing, density
  k-neighbors, freespace/occlusion switches and independent statistical filtering.
- Dynamic debugger capture of the active constructor: preset enum 0 (`standard`),
  resolution 0.01, normal limits 0.025/0.15, six levels and a 16-neighbor value.

### Point-cloud coloring — high confidence structure, medium confidence weights

Leaked compilation paths include:

- `color_extractors/patch-based/direct_patch_color_extractor.cpp`
- `patch_width_estimator.cpp`, `geometry/patch_projector.cpp`
- `colorizer/color_blending.cpp`, `color_extraction.cpp`, `colorizer.cpp`
- `optimal_view_selection.cpp`, `_voxelranking.cpp`, `_weight_maps.cpp`
- `depthmap/depthmap.cpp`, `_io.cpp`, `_rendering.cpp`, `raytracing/raycaster.hpp`
- `exposure_correction/global/global_exposure_optimizer.cpp`
- residuals for dynamic range, pairwise brightness, parameters, scene brightness and
  joint variance using a Ceres `GammaModel`
- Gaussian/mask-boundary weight maps, adaptive bandwidth selection and KNN/discard/fill
  painters.

Symbols establish depth-tested visibility, rolling-shutter projection, patch sampling,
voxel-ranked optimal views, five-view variants and weighted blending. Exact private score
curves are represented by monotonic incidence/distance/boundary equivalents.

### Image post-processing — high confidence operations

- Imports: LibRaw, DNG SDK, `GaussianBlur`, `medianBlur`, `fastNlMeansDenoising`,
  `createMergeMertens`, `addWeighted`, color conversion, threshold, resize/rotate and JPEG.
- CLI/options: high-quality/fast/plain; sharpening; person-region blur; DNG denoise modes
  adaptive/manual wavelet, NLM and fused NLM; exposure stops; auto/custom/camera/per-pano
  white balance; hidden HDR EV shifts and vignetting.
- A debugger capture at the active high-quality fused-NLM call recovered doubles `1.75`
  and `7.5`, template/search windows `7/17`, ISO 173 and reference exposure `1.5 EV`.
  Disassembly of `navvis::image::isoSensitivityToGain(unsigned)` in
  `liblibnavvis_image.so.4.85` gives `max(0, 4.5 + 6*log2(ISO/100))`. The caller computes
  `h = 1.75 * (7.5/1.75)^((gain + 6*reference_exposure)/27)`. For recorded ISOs
  173/147/122/123 this exactly predicts captured strengths
  4.6785717/4.3362451/3.9750445/3.9902132.
- Frozen intermediates establish three `0/1.5/3 EV` inputs and Mertens output bitwise;
  the NLM input is 99.999557% value-identical. The original JPEG stores the unrotated
  sensor raster and relies on EXIF Orientation, rather than physically rotating before
  encoding.

Four-camera decoded-JPEG MAE is 0.000670/0.000610/0.000499/0.000712 on an 8-bit scale.
This is pixel-level equivalence, not byte-identical JPEG encoding; private metadata beyond
orientation is not reconstructed.

### Panorama and point-cloud rendering — high confidence structure

Leaked paths identify `depth_map_optimizer.cpp`, `GaussNewtonDepthMapOptimizer`,
`blending_utils.cpp`, `multiband_blender.cpp`, `floor_filler.cpp`,
`exposure_compensator_soft_constraint.cpp`, `image_extrapolator.cpp`,
`image_stitcher.cpp`, seam utilities and `pyramid_inpainting.cpp`.

OpenCV imports prove Gaussian/Laplacian pyramid operations, distance transform, inpaint,
morphology and GraphCut/DP seam finders. Point-cloud renderer shader text describes rotating
a triangle by the point normal and scaling it to surfel radius, confirming oriented-disc
OpenGL splatting. Public defaults include width 8192, height 4096, radius 0.01 m, near 0.2 m
and far 35 m.

Runtime geometry probes recover the OCam row/column center convention, all four transverse
camera mappings (`x=-ray_camera.y`, `y=-ray_camera.x`), direct pose transpose and a
180-degree panorama-frame Y rotation. GDB captures additionally recover the standard G11
seam sequence (0-1, 1-2, 2-3, 0-3), circular shifts, 2048x1024 GraphCut canvas, terminal
cost 10000 and bad-region penalty 1000. The four final clean masks have IoU
99.10/99.84/99.52/98.41 percent against the captured binary masks. On the same binary-
projected inputs, the 2K panorama MAE is 4.4414/255. The standard dataset path never enters
`ExposureCompensatorSoftConstraint::computeGainValuesImpl`, so no exposure correction is the
evidence-backed default; the soft solver remains optional.

The binary no-floor image and binary valid mask provide a stage-isolated nadir test. The
8192x4096 missing-region MAE is 79.644/255 before filling and 2.885/255 after the reconstructed
frontier extrapolation/circular pyramid fill; full-image MAE is 0.4631/255 and bottom-20-percent
MAE is 2.293/255. Depth/operator mask generation, exact multiband rounding and small residual
nadir texture differences remain open.

### SLAM trajectory chain — complete result-level alignment

- Installed `surveyorslam_processing_node`, `compute_constraints`, `evaluate_constraints`
  and `compute_trajectories` identify the SurveyorSLAM/Cartographer-derived offline chain.
- The original log for `2026-02-08_07.33.20` records 6 submaps and 1,617 nodes. Constraint
  search considered 41 candidates and accepted 15; the graph contains 2,581 membership
  constraints, 15 loop constraints, 8,480 IMU samples, 1,617 pose variables and 6 submaps.
- The original upsampling-only phase reports 8,081 output poses and five Ceres iterations;
  the recorded input contains 1,545 online global and 1,599 local poses. Its private 5x
  timing/optimization is therefore not simple per-segment linear interpolation.

The clean reader and raw frontend now reproduce all 1,617 nodes, six submaps, every one of
the 2,581 memberships, 401 eligible loop pairs, 41 FixedRatioSampler searches and the exact
15 accepted loop pair set. The raw chain combines both Pandar sensors in the binary's 50 ms
window, uses raw IMU orientation and constant-velocity translation for per-return deskew,
then matches a 0.04 m scan cloud against 0.1/0.3/0.6 m float surfel levels. Local node-pose
translation error is mean/p95 0.497/0.732 mm; the only 4.123 mm maximum occurs at node 1150
on a submap boundary.

Native `HybridProbabilityGrid` implements the Cartographer probability lookup tables,
hit-before-miss ordering and sparse-cell storage. At the frozen `(submap 2,node 1120)` FCS
pose it reproduces the binary score `0.5435789227485657` exactly. A full binary `SAVE_ALL`
replay proves the 41-candidate outcome counts NONE/THRESH_ROT/HIGH_RES/ICP_STABILITY are
15/16/6/4 and that FCS is skipped when `max(search_region.scaling) < 0.1 m`. Frozen-cloud
loop measurement error is translation mean/p95/max 0.318/0.637/0.688 mm; generated-cloud
measurements are 0.817/1.218/1.243 mm while retaining the exact pair set.

The backend implements the binary-probed 15-parameter-block Exact factor. It groups 8,480
IMU samples into 1,616 adjacent-node 9D factors and optimizes pose, velocity, IMU extrinsics,
accelerometer/gyroscope intrinsics and gravity together with membership and loop factors.
With official topology/loops its node ATE translation mean/p95 is 0.139/0.337 mm. With the
entire generated frontend and generated loops, final node ATE translation mean/p95/max is
0.421/0.802/0.869 mm and rotation mean/p95 is 0.00213/0.00391 degrees; 1 s RPE translation
mean/p95 is 0.072/0.191 mm. Reference poses are used only after optimization for metrics.

This is result-level, not byte-level, alignment. Generated HybridGrid cell counts differ by
about 0.038 percent in total and the adaptive 5,000-point cloud still has container-order
differences. Exact evidence and standard commands are in
`../regression/2026-02-08_07.33.20/slam_raw/COMPLETE_SLAM_ALIGNMENT_20260825.md`,
`slam_probes/HYBRID_GRID_LOOP_EVIDENCE.md` and `slam_probes/ICP_STABILITY_EVIDENCE.md`.

### Floor estimator — full same-input and runner alignment

Compiled Cython module names identify floor clusters, refiner, splitter/merger, validation,
simple/refined estimators, trace reader/splitter and time-range merger. Runtime introspection
recovered these defaults: bin 0.1 m, standard floor 3.0 m, maximum 4.0 m, minimum 2.1 m,
tolerance 0.03 m, range gap 195,000,000 ns, validation period 100,000,000 ns, small segment
5,000,000,000 ns and overlap tolerance 0.12 m. Method names establish remove-empty,
merge-adjacent, merge-tiny and split-double-floor refinement.

Runtime proxying additionally established the sequential current/previous-floor state machine,
the exact significant-bin predicate, boundary-bin selection, incremental bin means, refiner order
and the full-trace seeded replay performed after a double-floor split. The recovered estimator
matches all 220 available reference `trace.csv`/`floors.json` pairs below
`/media/cybergeo/12T/DT`: 1,796,815 trace rows, 615 floors, zero mismatching JSON objects.

The runner trace path was separately recovered from the raw `/imu/magnetic_field` clock,
nanosecond trajectory interpolation, six-significant-digit serialization and Eigen quaternion
sign selection. `2026-07-21_11.07.05` (5,374 rows, one floor) and
`2026-01-19_19.04.51` (17,878 rows, five floors) reproduce the native `trace.csv` byte for byte
and the native `floors.json` field for field. Full evidence and commands are in
`FLOOR_FULL_ALIGNMENT_20260828.md`.

### Mapped-space quality — high confidence

Dynamic symbols expose `QualityVoxelGridAggregatorHash::addAlongRay`, `merge`,
`rescaleRayCounts` and `filterByRayCountAndCompact`. Public defaults are 1/6 m cells,
minimum 36 rays per voxel, 50 m maximum ray and 1000 m output tile length. The reference
implements the confirmed ray-count quantity; NavVis binary/Brotli serialization is omitted.

## Complete reference execution

The initially interrupted command was rerun from the unchanged recording into the isolated
`datasets_proc_reference` base directory. It completed successfully in 11,770.47 seconds
(3:16:25). The final reference contains 159,147,139 colored vertices, 299 panoramas at
8192×4096 and 1,196 processed camera JPEGs at 5472×3648 display orientation.

The reference cloud record is 36 bytes in this order: XYZ float32, RGBA uint8, intensity
float32, normal XYZ float32 and curvature float32. Its bounding box is
`[-42.589901, -42.706520, -8.228365]` to `[82.879669, 56.433723, 14.101768]` m.
Runtime logs from `nv_colorcloud` report 152,664,107 directly colored and 6,483,032
extrapolated points. The geometry stage reports 1.539 billion raw returns, a free-space
octree reduction from 181,325,866 to 141,455,016 leaves, and 1,447 output tiles.

This complete run supersedes the earlier interrupted-log qualification. Static evidence is
still used to identify implementation structure; numerical alignment claims below use the
completed output directly.

## G10/VLP16 compatibility execution

The G10-512 recording `2023-05-15_10.18.42` contains standard 1206-byte Velodyne VLP16
packets rather than the G11 recording's 820-byte Pandar packets. A full real-data run decoded
104,340 structurally valid packets, retained endpoint/origin ray history, used 1,385 supplied
SLAM poses over 69.25 seconds, generated four 8192x4096 panoramas, and produced a conservative
3,538,026-point colored cloud. All output coordinates and normals are finite and all normals
are unit length to float tolerance.

An A/B reconstruction from the identical raw shards produced 5,822,223 points with carving
disabled. The selected G10 thresholds (minimum six traversals and traversal/hit ratio 3.0)
remove 7,221,231 free-space-conflicting voxel records before density and surface filtering;
at five trajectory anchors they retain 63.22% to 72.40% of the no-carve surface within 2 cm.
The more aggressive G11 defaults retained only 49.10% to 59.60% and were therefore not used
as the G10 default.

The installed original pipeline could not provide a same-recording reference: its LicenseCheck
reported the G10-512 license expired and stopped before point-cloud processing. No license
bypass was attempted. Consequently this run proves compatibility and internal consistency,
not numerical equality with an original G10 output.

## Confidence legend

- **High**: named RTTI/symbol/import/source path, public help, direct constructor memory or
  recovered call order.
- **Medium**: algorithm family is proven but private weighting/threshold mapping is stripped;
  a clean equivalent is supplied.
- **External** in `pipeline.py`: orchestration, serialization, format conversion or a component
  for which no responsible algorithm was inferred.

## C++ reconstruction build validation

The `cpp/` tree contains 4,777 lines of C++17 headers, implementations, real-data workers and
smoke tests for the five native algorithm groups. It was configured and built locally with GCC 11.4,
CMake 3.22, Eigen 3.4, OpenCV 4.5.4 and OpenMP 4.5. The resulting static library links all
five modules. `ctest --output-on-failure` reports 1/1 test executable passed; that executable
checks five paths:

1. per-point trajectory interpolation/unskew in the cloud builder;
2. voxel aggregation and multi-scale planar normal recovery;
3. depth-tested patch color extraction and RGB output;
4. exposure/image processing and JPEG encoding;
5. panorama stitching and surfel depth rendering.

PCL and Ceres development headers are not installed on this host. Their proven original
roles are implemented with Eigen/OpenCV equivalents so the delivered source is genuinely
buildable rather than pseudocode. The class and method names observed in the binaries are
preserved where evidence supports them.

## Real rec-v4 adapter and full-alignment validation

The current unified runner was exercised with the supplied 185,621-pose reference trajectory,
0.5 seconds of both laser streams and one 512×256 panorama. It consumed 3,000 packet records,
retained 64 raw shards, reconstructed 120,315 surface vertices, emitted processed camera JPEGs,
performed direct-camera depth-tested coloring and completed in 7.6 seconds. The generated
`cam_head` and four camera poses for capture 00000 match the completed reference info at the
printed precision (0 cm and 0 degrees).

The complete C++ laser run decoded all 33 horizontal plus 33 vertical bags into 1,192,349,760
endpoint/origin-cluster records and 502,884,086 occupied 1 cm voxels. The retained 56-byte
`.raytile` records carry the world-space beam origin in addition to endpoint, normal and
intensity statistics. Sparse clipped 3D-DDA ray carving rejected 24,933,821 free-space-conflict
candidates before density filtering; final output voxelization produced 151,301,323 vertices,
-4.93% from the reference count.

Across five exact trajectory-centered regions, reference-to-candidate median distances are
0.58, 0.63, 0.61, 0.60 and 0.81 cm; 96.00–99.65% of reference vertices have a candidate within
2 cm. Candidate-to-reference medians are 0.39, 0.42, 0.44, 0.41 and 1.01 cm, with 76.31–94.31%
within 2 cm. Relative to the endpoint-only v4 result, mean five-region candidate-to-reference
distance, p90 and p99 improve by 42.45%, 23.26% and 59.61%; mean 2 cm precision rises 4.60
percentage points while mean reference coverage changes by -0.13 points. This validates the
missing ray-history fix without claiming the proprietary octree thresholds are identical.

All 299 reconstructed panoramas have the native 8192×4096 reference shape. At 1024×512, the
all-capture mean/median RGB MAE is 22.14/21.77 on an 8-bit scale, mean grayscale SSIM is
0.534, and median absolute phase shifts are 0.47 px horizontal and 0.74 px vertical. The
first independently processed camera image has 12.65/255 MAE and approximately 0.004 px phase
shift against the corresponding reference JPEG.

The final colored cloud contains all 151,301,323 surface vertices. Direct depth-tested camera
fusion colored 134,799,041 vertices (89.09%); panorama fallback colored 16,502,282 (10.91%).
Across the 11 whole-cloud RGB quantiles, every candidate channel value is within one 8-bit level
of the reference. For geometrically matched vertices within 1 cm in the five validation regions,
RGB MAE is 34.31, 32.29, 38.97, 22.41 and 46.86/255. This distinguishes strong global tone
alignment from the still-measurable local patch/occlusion mismatch.
