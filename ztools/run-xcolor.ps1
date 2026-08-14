# PowerShell script for xsfm processing pipeline

# Accept data directory path as parameter
param(
  [Parameter(Mandatory=$true, HelpMessage="Path to the data directory")]
  [string]$dataDir,

  [Parameter(Mandatory=$false, HelpMessage="Path to the FAISS vocabulary tree")]
  [string]$vocabTreePath = "",

  [Parameter(Mandatory=$false, HelpMessage="Point cloud LAS file; defaults to dataDir\map_opt.las")]
  [string]$PointCloudPath = "",

  [Parameter(Mandatory=$false, HelpMessage="Camera directories to delete and exclude, for example cam2")]
  [string[]]$ExcludeCamera = @(),

  [Parameter(Mandatory=$false, HelpMessage="Relax triangulation and filtering only when the initial poses are known to be inaccurate")]
  [switch]$InaccurateInitialPose
)

# Set tool directories
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$BUILD_PACK = Join-Path $PROJECT_ROOT "build-pack"

# Runtime executables must come from the distribution assembled by
# ztools/build-pack.ps1. Do not fall back to development build directories.
$xsfm_pc_exe = Join-Path $BUILD_PACK "xsfm_process_point_cloud.exe"
$xsfm_pre_exe = Join-Path $BUILD_PACK "xsfm_pre.exe"
$xsfm_post_exe = Join-Path $BUILD_PACK "xsfm_post.exe"
$xcolor_main_exe = Join-Path $BUILD_PACK "xcolor.exe"
$createInitialModelExe = Join-Path $BUILD_PACK "xsfm_create_initial_model.exe"

if ([string]::IsNullOrWhiteSpace($vocabTreePath)) {
  $vocabCandidates = @(
    (Join-Path $BUILD_PACK "vocab_tree_faiss_flickr100K_words32K.bin"),
    (Join-Path (Split-Path -Parent $PROJECT_ROOT) "reality-flow\resources\xsfm\vocab_tree_faiss_flickr100K_words32K.bin")
  )
  $vocabTreePath = $vocabCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

Write-Host "Build pack dir: $BUILD_PACK" -ForegroundColor Gray
Write-Host "xsfm point cloud: $xsfm_pc_exe" -ForegroundColor Gray
Write-Host "xsfm pre:         $xsfm_pre_exe" -ForegroundColor Gray
Write-Host "xsfm post:        $xsfm_post_exe" -ForegroundColor Gray
Write-Host "xcolor:           $xcolor_main_exe" -ForegroundColor Gray
Write-Host "initial model:    $createInitialModelExe" -ForegroundColor Gray
Write-Host "vocabulary tree:  $vocabTreePath" -ForegroundColor Gray
Write-Host "inaccurate pose:  $InaccurateInitialPose" -ForegroundColor Gray

foreach ($tool in @(
    $xsfm_pc_exe,
    $xsfm_pre_exe,
    $xsfm_post_exe,
    $xcolor_main_exe,
    $createInitialModelExe
  )) {
  if ([string]::IsNullOrWhiteSpace($tool) -or -not (Test-Path -LiteralPath $tool)) {
    Write-Host "ERROR: Required packaged tool was not found: $tool" -ForegroundColor Red
    Write-Host "Run .\ztools\build-pack.ps1 to assemble all runtime executables." -ForegroundColor Yellow
    exit 1
  }
}

# Validate parameter
if (-not (Test-Path $dataDir)) {
  Write-Host "ERROR: Data directory not found: $dataDir" -ForegroundColor Red
  Write-Host "Usage: .\run-xcolor.ps1 'C:\path\to\data'" -ForegroundColor Yellow
  exit 1
}

$dataDir = (Resolve-Path -LiteralPath $dataDir).Path
$imagesRoot = Join-Path $dataDir "images"
if (-not (Test-Path -LiteralPath $imagesRoot -PathType Container)) {
  Write-Host "ERROR: Images directory not found: $imagesRoot" -ForegroundColor Red
  exit 1
}
$imagesRoot = (Resolve-Path -LiteralPath $imagesRoot).Path

foreach ($cameraName in $ExcludeCamera) {
  if ($cameraName -notmatch '^cam\d+$') {
    Write-Host "ERROR: Invalid camera directory name: $cameraName" -ForegroundColor Red
    exit 1
  }

  $cameraDir = Join-Path $imagesRoot $cameraName
  if (-not (Test-Path -LiteralPath $cameraDir)) {
    Write-Host "Excluded camera is already absent: $cameraName" -ForegroundColor Yellow
    continue
  }

  $resolvedCameraDir = (Resolve-Path -LiteralPath $cameraDir).Path
  if (-not $resolvedCameraDir.StartsWith(
      $imagesRoot + [IO.Path]::DirectorySeparatorChar,
      [StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "ERROR: Refusing to delete camera outside images directory: $resolvedCameraDir" -ForegroundColor Red
    exit 1
  }

  $imageCount = @(Get-ChildItem -LiteralPath $resolvedCameraDir -File).Count
  Remove-Item -LiteralPath $resolvedCameraDir -Recurse -Force
  Write-Host "Deleted excluded camera: $resolvedCameraDir ($imageCount files)" -ForegroundColor Yellow
}

$xsfmDir = Join-Path $dataDir "xsfm"
if (Test-Path -LiteralPath $xsfmDir) {
  $resolvedXsfmDir = (Resolve-Path -LiteralPath $xsfmDir).Path
  if (-not $resolvedXsfmDir.StartsWith(
      $dataDir + [IO.Path]::DirectorySeparatorChar,
      [StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "ERROR: Refusing to clear xsfm outside data directory: $resolvedXsfmDir" -ForegroundColor Red
    exit 1
  }
  Remove-Item -LiteralPath $resolvedXsfmDir -Recurse -Force
}
New-Item $xsfmDir -Type Directory -Force | Out-Null

if ([string]::IsNullOrWhiteSpace($vocabTreePath) -or -not (Test-Path -LiteralPath $vocabTreePath)) {
  Write-Host "ERROR: Vocabulary tree was not found. Pass -vocabTreePath explicitly." -ForegroundColor Red
  exit 1
}

# Verify required files exist
Write-Host "Checking required files..." -ForegroundColor Cyan
$lasFile = if ([string]::IsNullOrWhiteSpace($PointCloudPath)) {
  Join-Path $dataDir "map_opt.las"
} else {
  $PointCloudPath
}
$poseFile = "$dataDir\images\ImgPose.txt"

Write-Host "Looking for: $lasFile" -ForegroundColor Yellow
Write-Host "Looking for: $poseFile" -ForegroundColor Yellow

if (-not (Test-Path $lasFile)) {
  Write-Host "ERROR: LAS file not found at: $lasFile" -ForegroundColor Red
  Write-Host "Available files in directory:" -ForegroundColor Yellow
  Get-ChildItem $dataDir -Recurse -Filter "*.las" 2>/dev/null | ForEach-Object { Write-Host "  $_" }
  exit 1
}
$lasFile = (Resolve-Path -LiteralPath $lasFile).Path

if (-not (Test-Path $poseFile)) {
  Write-Host "ERROR: Pose file not found at: $poseFile" -ForegroundColor Red
  exit 1
}

$imagesDir = Join-Path $dataDir "images"
$activeCameraNames = @(
  Get-ChildItem -LiteralPath $imagesDir -Directory |
    Where-Object { $_.Name -match '^cam\d+$' } |
    Sort-Object Name |
    Select-Object -ExpandProperty Name
)

if ($activeCameraNames.Count -eq 0) {
  Write-Host "ERROR: No camN image directories found under $imagesDir" -ForegroundColor Red
  exit 1
}

$activePoseFile = Join-Path $dataDir "xsfm\ImgPose.active.txt"
$poseLines = @(Get-Content -LiteralPath $poseFile)
$activePoseLines = @($poseLines[0])
$activePoseLines += @(
  $poseLines | Select-Object -Skip 1 | Where-Object {
    $imageName = ($_ -split '\s+')[0]
    $cameraName = ($imageName -split '/')[0]
    $activeCameraNames -contains $cameraName
  }
)
$activePoseLines | Set-Content -LiteralPath $activePoseFile -Encoding utf8

$calibrationFile = Join-Path $imagesDir "rig.json"
if (-not (Test-Path -LiteralPath $calibrationFile)) {
  Write-Host "ERROR: Camera calibration file not found at: $calibrationFile" -ForegroundColor Red
  exit 1
}

$calibrationDocument = @(Get-Content -LiteralPath $calibrationFile -Raw | ConvertFrom-Json)
$cameraConfigs = @(
  $calibrationDocument[0].cameras | Where-Object {
    $activeCameraNames -contains $_.image_prefix.TrimEnd('/')
  }
)
if ($cameraConfigs.Count -ne $activeCameraNames.Count) {
  Write-Host "ERROR: Missing independent camera calibration for one or more active cameras." -ForegroundColor Red
  exit 1
}

Write-Host "Active cameras: $($activeCameraNames -join ', ')" -ForegroundColor Green
Write-Host "Active poses:   $($activePoseLines.Count - 1)" -ForegroundColor Green

Write-Host "Files verified successfully!" -ForegroundColor Green

# Step 1: Process point cloud with xsfm_process_point_cloud.exe
Write-Host "Step 1: Processing point cloud..." -ForegroundColor Green
& $xsfm_pc_exe `
  --las_filename "$lasFile" `
  --initial_pose_filename "$activePoseFile" `
  --output_dir "$dataDir\xsfm" `
  --nooutput_full `
  --output_normals_pcd

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 1" -ForegroundColor Red
  exit 1
}

# Step 2: Extract features with xsfm_pre.exe
$dbFile = "$dataDir\xsfm\xsfm.db"
Write-Host "Step 2: Extracting features for independent cameras..." -ForegroundColor Green
foreach ($cameraConfig in $cameraConfigs) {
  $cameraName = $cameraConfig.image_prefix.TrimEnd('/')
  $cameraImageList = Join-Path $xsfmDir "$cameraName-images.txt"
  @(
    Get-ChildItem -LiteralPath (Join-Path $imagesDir $cameraName) -File -Filter "*.jpg" |
      Sort-Object Name |
      ForEach-Object { "$cameraName/$($_.Name)" }
  ) | Set-Content -LiteralPath $cameraImageList -Encoding utf8

  $cameraParams = @($cameraConfig.camera_params) -join ','
  Write-Host "  ${cameraName}: $cameraParams" -ForegroundColor Gray
  & $xsfm_pre_exe feature_extractor `
    --image_path "$imagesDir" `
    --image_list_path "$cameraImageList" `
    --database_path "$dbFile" `
    --ImageReader.camera_model "$($cameraConfig.camera_model_name)" `
    --ImageReader.camera_params "$cameraParams" `
    --ImageReader.single_camera 1 `
    --SiftExtraction.max_num_features 5000

  if ($LASTEXITCODE -ne 0) {
    Write-Host "Error extracting features for $cameraName" -ForegroundColor Red
    exit 1
  }
}

# Step 3: Sequential matching with xsfm_pre.exe
Write-Host "Step 3: Sequential matching..." -ForegroundColor Green
& $xsfm_pre_exe sequential_matcher `
  --database_path "$dataDir\xsfm\xsfm.db" `
  --SequentialMatching.vocab_tree_path "$vocabTreePath" `
  --SequentialMatching.loop_detection 1 `
  --SequentialMatching.quadratic_overlap 0 `
  --TwoViewGeometry.filter_stationary_matches 1

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 3" -ForegroundColor Red
  exit 1
}

# Step 4: Build the registered camera model directly from the initial poses,
# then triangulate and optimize with initial-position constraints only.
Write-Host "Step 4: Running initial-pose constrained point refiner..." -ForegroundColor Green
$sfmOutputDir = "$dataDir\xsfm\sparse"
$initialModelDir = "$dataDir\xsfm\initial_model"
New-Item $sfmOutputDir -Type Directory -Force | Out-Null
New-Item $initialModelDir -Type Directory -Force | Out-Null

$resolvedDataDir = (Resolve-Path $dataDir).Path
$resolvedSfmOutputDir = (Resolve-Path $sfmOutputDir).Path
if (-not $resolvedSfmOutputDir.StartsWith($resolvedDataDir, [System.StringComparison]::OrdinalIgnoreCase)) {
  Write-Host "ERROR: Refusing to clear output outside data directory: $sfmOutputDir" -ForegroundColor Red
  exit 1
}

Get-ChildItem -LiteralPath $sfmOutputDir -Force | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $initialModelDir -Force | Remove-Item -Recurse -Force

& $createInitialModelExe `
  --database_path "$dbFile" `
  --pose_path "$dataDir\xsfm\localenu_pose.txt" `
  --output_path "$initialModelDir"

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 4 (create initial pose model)" -ForegroundColor Red
  exit 1
}

$refinedModelDir = Join-Path $sfmOutputDir "0"
New-Item $refinedModelDir -Type Directory -Force | Out-Null
$refinerArgs = @(
  "--database_path", $dbFile,
  "--image_path", $imagesDir,
  "--input_path", $initialModelDir,
  "--output_path", $refinedModelDir,
  "--num_rounds", "3",
  "--clear_points", "1",
  "--use_pose_prior", "1",
  "--pose_prior_position_stddev", "0.1",
  "--use_robust_loss_on_pose_prior", "1",
  "--pose_prior_loss_scale", "2.795",
  "--use_point_cloud_constraint", "0",
  "--Mapper.ba_refine_focal_length", "0",
  "--Mapper.ba_refine_principal_point", "0",
  "--Mapper.ba_refine_extra_params", "0",
  "--Mapper.ba_refine_sensor_from_rig", "0",
  "--Mapper.ba_use_gpu", "1",
  "--BundleAdjustment.refine_focal_length", "1",
  "--BundleAdjustment.refine_principal_point", "1",
  "--BundleAdjustment.refine_extra_params", "1",
  "--BundleAdjustment.refine_rig_from_world", "1",
  "--BundleAdjustment.refine_sensor_from_rig", "1",
  "--BundleAdjustment.refine_points3D", "1",
  "--BundleAdjustmentCeres.max_num_iterations", "50",
  "--BundleAdjustmentCeres.use_gpu", "1",
  "--BundleAdjustment.min_track_length", "3"
)

if ($InaccurateInitialPose) {
  Write-Host "Using relaxed triangulation for inaccurate initial poses." -ForegroundColor Yellow
  $refinerArgs += @(
    "--Mapper.tri_create_max_angle_error", "3.0",
    "--Mapper.tri_continue_max_angle_error", "3.0",
    "--Mapper.tri_complete_max_reproj_error", "6.0",
    "--Mapper.tri_merge_max_reproj_error", "6.0",
    "--Mapper.tri_ignore_two_view_tracks", "0",
    "--Mapper.filter_max_reproj_error", "6.0",
    "--Mapper.filter_min_tri_angle", "1.0"
  )
}

& $xsfm_pre_exe constrained_point_refiner @refinerArgs

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 4 (LiDAR-constrained refinement)" -ForegroundColor Red
  exit 1
}

# Step 4.5: Convert fisheye SfM result to cubemap model and depth maps
Write-Host "Step 4.5: Running xsfm post..." -ForegroundColor Green
$xsfmPostOutputDir = "$dataDir\xsfm\cubemap_colmap"
$xsfmPostImagesPath = "$xsfmPostOutputDir\images"
$xsfmPostSfmPath = "$xsfmPostOutputDir\sparse"
$sfmModelDir = if (Test-Path -LiteralPath "$sfmOutputDir\0") {
  "$sfmOutputDir\0"
} elseif (Test-Path -LiteralPath "$sfmOutputDir\cameras.bin") {
  $sfmOutputDir
} else {
  $null
}

if (-not (Test-Path $xsfm_post_exe)) {
  Write-Host "ERROR: xsfm_post executable not found at: $xsfm_post_exe" -ForegroundColor Red
  exit 1
}

if ([string]::IsNullOrWhiteSpace($sfmModelDir)) {
  Write-Host "ERROR: SfM model not found under: $sfmOutputDir" -ForegroundColor Red
  exit 1
}

$xsfmPostArgs = @(
  "--model-dir", $sfmModelDir,
  "--image-dir", "$dataDir\images",
  "--output-dir", $xsfmPostOutputDir,
  "--point-cloud-path", "$dataDir\xsfm\localenu_normal.pcd",
  "--image-step", "2",
  "--depth-voxel-size", "0.03",
  "--overwrite"
)
$cameraMaskDir = Join-Path $BUILD_PACK "camera_masks"
if (-not (Test-Path -LiteralPath $cameraMaskDir -PathType Container)) {
  Write-Host "ERROR: Fixed camera mask directory not found: $cameraMaskDir" -ForegroundColor Red
  exit 1
}
foreach ($cameraName in $activeCameraNames) {
  $cameraMaskFile = Join-Path $cameraMaskDir "$cameraName.png"
  if (-not (Test-Path -LiteralPath $cameraMaskFile -PathType Leaf)) {
    Write-Host "ERROR: Fixed camera mask not found: $cameraMaskFile" -ForegroundColor Red
    exit 1
  }
}
Write-Host "Using packaged fixed camera masks: $cameraMaskDir" -ForegroundColor Green
$xsfmPostArgs += @("--mask-dir", $cameraMaskDir)

& $xsfm_post_exe @xsfmPostArgs

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 4.5 (xsfm post)" -ForegroundColor Red
  exit 1
}

# Step 5: Colorize point cloud with xcolor.exe
Write-Host "Step 5: Colorizing point cloud..." -ForegroundColor Green
# Determine the correct camera path (try multiple possible locations)
$images_path = $null
$possible_paths = @(
  "$xsfmPostImagesPath"
)

foreach ($path in $possible_paths) {
  if (Test-Path $path) {
    $images_path = $path
    break
  }
}

if (-not $images_path) {
  Write-Host "ERROR: Could not find images directory" -ForegroundColor Red
  Write-Host "Searched in: $($possible_paths -join ', ')" -ForegroundColor Yellow
  exit 1
}

& $xcolor_main_exe `
  --images_path "$images_path" `
  --sfm_result_path "$xsfmPostSfmPath" `
  --point_cloud_filename "$lasFile" `
  --output_path "$dataDir"

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 5" -ForegroundColor Red
  exit 1
}

Write-Host "Point cloud colorization completed!" -ForegroundColor Green

Write-Host "Pipeline completed successfully!" -ForegroundColor Green
