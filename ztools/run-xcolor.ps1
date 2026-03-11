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
$BUILD_DIR = Join-Path $PROJECT_ROOT "build\RelWithDebInfo"
$NATIVE_TOOLS = Join-Path $PROJECT_ROOT "native-tools"

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
$lasFile = "$dataDir\map.las"
$poseFile = "$dataDir\camera\ImgPose.txt"

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
  --las_filename "$dataDir\map.las" `
  --initial_pose_filename "$dataDir\camera\ImgPose.txt" `
  --output_dir "$dataDir\xsfm" `
  --nooutput_full

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 1" -ForegroundColor Red
  exit 1
}

# Step 2: Extract features with xsfm_pre.exe
Write-Host "Step 2: Extracting features..." -ForegroundColor Green
$xsfm_pre_exe = Join-Path $BUILD_PACK "xsfm_pre.exe"
& $xsfm_pre_exe feature_extractor `
  --image_path "$dataDir\camera" `
  --database_path "$dataDir\xsfm\xsfm.db" `
  --ImageReader.camera_model OPENCV_FISHEYE `
  --ImageReader.camera_params "1029.5272863732237,1032.9740976263085,1920,1920,0.037416683931696879,-0.0051502247212099643,0.0064400644003639101,-0.002301772325582836" `
  --ImageReader.single_camera_per_folder 1 `
  --SiftExtraction.max_num_features 8000

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 2" -ForegroundColor Red
  exit 1
}

# Step 3: Sequential matching with xsfm_pre.exe
Write-Host "Step 3: Sequential matching..." -ForegroundColor Green
& $xsfm_pre_exe sequential_matcher `
  --database_path "$dataDir\xsfm\xsfm.db" `
  --SequentialMatching.vocab_tree_path "$BUILD_DIR\vocab_tree_faiss_flickr100K_words32K.bin" `
  --SequentialMatching.loop_detection 1 `
  --SequentialMatching.loop_detection_period 1 `
  --SequentialMatching.loop_detection_num_nearest_neighbors 4 `
  --SequentialMatching.loop_detection_num_images 70 `
  --TwoViewGeometry.filter_stationary_matches 1

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 3" -ForegroundColor Red
  exit 1
}

# Step 4: Run xsfm main pipeline
Write-Host "Step 4: Running main xsfm pipeline..." -ForegroundColor Green
$xsfm_exe = Join-Path $BUILD_PACK "xsfm.exe"
& $xsfm_exe `
  -point_cloud_filename "$dataDir\xsfm\localenu.pcd" `
  -point_cloud_offset_filename "$dataDir\xsfm\localenu.json" `
  -database_filename "$dataDir\xsfm\xsfm.db" `
  -initial_pose_filename "$dataDir\xsfm\localenu_pose.txt" `
  -images_path "$dataDir\0\camera" `
  -output_path "$dataDir\xsfm\sparse" `
  --use_point_cloud

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 4" -ForegroundColor Red
  exit 1
}

# Step 5: Colorize point cloud with xcolor_main.exe
Write-Host "Step 5: Colorizing point cloud..." -ForegroundColor Green
$xcolor_main_exe = Join-Path $BUILD_PACK "xcolor_main.exe"

# Determine the correct camera path (try multiple possible locations)
$images_path = $null
$possible_paths = @(
  "$dataDir\camera",
  "$dataDir\0\camera"
)

foreach ($path in $possible_paths) {
  if (Test-Path $path) {
    $images_path = $path
    break
  }
}

if (-not $images_path) {
  Write-Host "ERROR: Could not find camera images directory" -ForegroundColor Red
  Write-Host "Searched in: $($possible_paths -join ', ')" -ForegroundColor Yellow
  exit 1
}

& $xcolor_main_exe `
  --images_path "$images_path" `
  --sfm_result_path "$dataDir\xsfm\sparse\0" `
  --point_cloud_filename "$dataDir\map.las" `
  --output_path "$dataDir"

if ($LASTEXITCODE -ne 0) {
  Write-Host "Error in Step 5" -ForegroundColor Red
  exit 1
}

Write-Host "Point cloud colorization completed!" -ForegroundColor Green

Write-Host "Pipeline completed successfully!" -ForegroundColor Green
