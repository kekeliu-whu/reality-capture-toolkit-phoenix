# PowerShell script for xsfm processing pipeline

# Accept data directory path as parameter
param(
  [Parameter(Mandatory=$true, HelpMessage="Path to the data directory")]
  [string]$dataDir
)

# Set tool directories
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$BUILD_PACK = Join-Path $PROJECT_ROOT "build-pack"
$NATIVE_TOOLS = Join-Path $PROJECT_ROOT "native-tools"
$xsfm_pre_exe = Join-Path $BUILD_PACK "xsfm_pre_new.exe"
$xsfm_post_exe = Join-Path $BUILD_PACK "xsfm_post.exe"
$PYTHON_TOOLS_DEV = Join-Path $PROJECT_ROOT "build-all\build-python-tools"
$TOOL_SEARCH_DIRS = @($BUILD_PACK, $PYTHON_TOOLS_DEV)

function Find-ToolExecutable {
  param(
    [Parameter(Mandatory=$true)]
    [string]$Name
  )

  foreach ($dir in $TOOL_SEARCH_DIRS) {
    $candidate = Join-Path $dir $Name
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  return $null
}

$fix_rig_exe = Find-ToolExecutable "xsfm_fix_rig_database.exe"
$inject_priors_exe = Find-ToolExecutable "xsfm_inject_subview_priors.exe"

Write-Host "Build pack dir: $BUILD_PACK" -ForegroundColor Gray

# Validate parameter
if (-not (Test-Path $dataDir)) {
  Write-Host "ERROR: Data directory not found: $dataDir" -ForegroundColor Red
  Write-Host "Usage: .\run-xcolor.ps1 'C:\path\to\data'" -ForegroundColor Yellow
  exit 1
}

New-Item "$dataDir\xsfm" -Type Directory -Force | Out-Null

# Verify required files exist
Write-Host "Checking required files..." -ForegroundColor Cyan
$lasFile = "$dataDir\map_opt.las"
$poseFile = "$dataDir\images\ImgPose.txt"

Write-Host "Looking for: $lasFile" -ForegroundColor Yellow
Write-Host "Looking for: $poseFile" -ForegroundColor Yellow

if (-not (Test-Path $lasFile)) {
  Write-Host "ERROR: LAS file not found at: $lasFile" -ForegroundColor Red
  Write-Host "Available files in directory:" -ForegroundColor Yellow
  Get-ChildItem $dataDir -Recurse -Filter "*.las" 2>/dev/null | ForEach-Object { Write-Host "  $_" }
  exit 1
}

if (-not (Test-Path $poseFile)) {
  Write-Host "ERROR: Pose file not found at: $poseFile" -ForegroundColor Red
  exit 1
}

Write-Host "Files verified successfully!" -ForegroundColor Green

# Step 1: Process point cloud with xsfm_process_point_cloud.exe
Write-Host "Step 1: Processing point cloud..." -ForegroundColor Green
$xsfm_pc_exe = Join-Path $BUILD_PACK "xsfm_process_point_cloud.exe"
& $xsfm_pc_exe `
  --las_filename "$dataDir\map_opt.las" `
  --initial_pose_filename "$dataDir\images\ImgPose.txt" `
  --output_dir "$dataDir\xsfm" `
  --nooutput_full `
  --output_normals_pcd

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 1" -ForegroundColor Red
  exit 1
}

# Step 2: Extract features with xsfm_pre_new.exe
Write-Host "Step 2: Extracting features..." -ForegroundColor Green
& $xsfm_pre_exe feature_extractor `
  --image_path "$dataDir\images" `
  --database_path "$dataDir\xsfm\xsfm.db" `
  --ImageReader.camera_model OPENCV_FISHEYE `
  --ImageReader.camera_params "1030.1467091090503,1028.8008312807606,1949.5400400646813,1914.7835423220931,0.042605851463874037,-0.007532224456171745,0.0070007420682475438,-0.0023540347797643482" `
  --ImageReader.single_camera_per_folder 1 `
  --SiftExtraction.max_num_features 8000

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 2" -ForegroundColor Red
  exit 1
}

# Step 2.5: Fix rig structure in COLMAP database
Write-Host "Step 2.5: Fixing rig database..." -ForegroundColor Green
$dbFile = "$dataDir\xsfm\xsfm.db"
$rigFile = "$dataDir\images\rig.json"

if (-not (Test-Path $dbFile)) {
  Write-Host "ERROR: Database file not found at: $dbFile" -ForegroundColor Red
  exit 1
}

if (-not (Test-Path $rigFile)) {
  Write-Host "ERROR: Rig file not found at: $rigFile" -ForegroundColor Red
  Write-Host "Please provide rig.json under images directory." -ForegroundColor Yellow
  exit 1
}

if ([string]::IsNullOrEmpty($fix_rig_exe) -or -not (Test-Path $fix_rig_exe)) {
  Write-Host "ERROR: Rig database fix executable not found." -ForegroundColor Red
  Write-Host "Searched in: $($TOOL_SEARCH_DIRS -join ', ')" -ForegroundColor Yellow
  exit 1
}

# & $xsfm_pre_exe rig_configurator `
#   --database_path "$dbFile" `
#   --rig_config_path "$rigFile"

& $fix_rig_exe `
  --database_path "$dbFile" `
  --rig_config "$rigFile" `
  --backup

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 2.5 (fix rig database)" -ForegroundColor Red
  exit 1
}

# Step 2.6: Inject initial pose priors into COLMAP database
Write-Host "Step 2.6: Injecting initial pose priors..." -ForegroundColor Green
if ([string]::IsNullOrEmpty($inject_priors_exe) -or -not (Test-Path $inject_priors_exe)) {
  Write-Host "ERROR: Pose prior injection executable not found." -ForegroundColor Red
  Write-Host "Searched in: $($TOOL_SEARCH_DIRS -join ', ')" -ForegroundColor Yellow
  exit 1
}

& $inject_priors_exe `
  --database_path "$dbFile" `
  --trajectory_path "$poseFile" `
  --num_cameras 2 `
  --camera_prefix cam `
  --stddev 0.05 `
  --coordinate_system 1 `
  --ref_camera_only

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 2.6 (inject pose priors)" -ForegroundColor Red
  exit 1
}

# Step 3: Sequential matching with xsfm_pre.exe
Write-Host "Step 3: Sequential matching..." -ForegroundColor Green
& $xsfm_pre_exe sequential_matcher `
  --database_path "$dataDir\xsfm\xsfm.db" `
  --SequentialMatching.overlap 20 `
  --SequentialMatching.quadratic_overlap 0 `
  --SequentialMatching.loop_detection 0

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 3 (overlap matcher)" -ForegroundColor Red
  exit 1
}

& $xsfm_pre_exe sequential_matcher `
  --database_path "$dataDir\xsfm\xsfm.db" `
  --SequentialMatching.vocab_tree_path "$BUILD_PACK\vocab_tree_faiss_flickr100K_words32K.bin" `
  --SequentialMatching.loop_detection 1 `
  --SequentialMatching.loop_detection_period 8 `
  --TwoViewGeometry.filter_stationary_matches 1

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 3" -ForegroundColor Red
  exit 1
}

# Step 4: Run MGSfM global mapper
Write-Host "Step 4: Running MGSfM global mapper..." -ForegroundColor Green
$sfmOutputDir = "$dataDir\xsfm\sparse"
New-Item $sfmOutputDir -Type Directory -Force | Out-Null

$resolvedDataDir = (Resolve-Path $dataDir).Path
$resolvedSfmOutputDir = (Resolve-Path $sfmOutputDir).Path
if (-not $resolvedSfmOutputDir.StartsWith($resolvedDataDir, [System.StringComparison]::OrdinalIgnoreCase)) {
  Write-Host "ERROR: Refusing to clear output outside data directory: $sfmOutputDir" -ForegroundColor Red
  exit 1
}

Get-ChildItem -LiteralPath $sfmOutputDir -Force | Remove-Item -Recurse -Force

& $xsfm_pre_exe global_mapper `
  --database_path "$dataDir\xsfm\xsfm.db" `
  --image_path "$dataDir\images" `
  --output_path "$sfmOutputDir" `
  --GlobalMapper.use_multi_camera_pipeline 1 `
  --GlobalMapper.ba_refine_focal_length 0 `
  --GlobalMapper.ba_refine_principal_point 0 `
  --GlobalMapper.ba_refine_extra_params 0 `
  --GlobalMapper.use_prior_position 1 `
  --GlobalMapper.gp_prior_position_weight 1.0 `
  --GlobalMapper.pcd_path "$dataDir\xsfm\localenu_normal.pcd" `
  --GlobalMapper.pcd_max_camera_distance 20 `
  --GlobalMapper.pcd_max_distance 1.0 `
  --GlobalMapper.pcd_proj_weight 0.1 `
  --GlobalMapper.pcd_icp_weight 2 `
  --GlobalMapper.pcd_icp_ground_weight 2 `
  --GlobalMapper.pcd_huber_threshold 0.1 `
  --GlobalMapper.pcd_rematch_iterations 2 `
  --GlobalMapper.pcd_filter_los 1

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 4" -ForegroundColor Red
  exit 1
}

# Step 4.5: Convert fisheye SfM result to cubemap model and depth maps
Write-Host "Step 4.5: Running xsfm post..." -ForegroundColor Green
$xsfmPostOutputDir = "$dataDir\xsfm\cubemap_colmap"
$xsfmPostImagesPath = "$xsfmPostOutputDir\images"
$xsfmPostSfmPath = "$xsfmPostOutputDir\sparse"

if (-not (Test-Path $xsfm_post_exe)) {
  Write-Host "ERROR: xsfm_post executable not found at: $xsfm_post_exe" -ForegroundColor Red
  exit 1
}

if (-not (Test-Path "$sfmOutputDir\0")) {
  Write-Host "ERROR: SfM model not found at: $sfmOutputDir\0" -ForegroundColor Red
  exit 1
}

& $xsfm_post_exe `
  --model-dir "$sfmOutputDir\0" `
  --image-dir "$dataDir\images" `
  --output-dir "$xsfmPostOutputDir" `
  --point-cloud-path "$dataDir\xsfm\localenu_normal.pcd" `
  --overwrite

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 4.5 (xsfm post)" -ForegroundColor Red
  exit 1
}

# Step 5: Colorize point cloud with xcolor.exe
Write-Host "Step 5: Colorizing point cloud..." -ForegroundColor Green
$xcolor_main_exe = Join-Path $BUILD_PACK "xcolor.exe"

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
  --point_cloud_filename "$dataDir\map_opt.las" `
  --output_path "$dataDir"

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 5" -ForegroundColor Red
  exit 1
}

Write-Host "Point cloud colorization completed!" -ForegroundColor Green

Write-Host "Pipeline completed successfully!" -ForegroundColor Green
