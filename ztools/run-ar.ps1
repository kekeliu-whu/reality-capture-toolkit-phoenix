#!/usr/bin/env pwsh
# Run Insta360 AR Pipeline
# Usage: .\run-ar.ps1 -inputdir <dir> -insvpath <video.insv> -outputdir <dir> [-calibfile <calib.dat>] [-movpath <video.mov>] [-trajectoryfile <traj.txt>]

param(
    [Parameter(Mandatory=$true)]
    [string]$inputdir,
    
    [Parameter(Mandatory=$true)]
    [string]$insvpath,

    [string]$movpath,
    
    [Parameter(Mandatory=$true)]
    [string]$outputdir,
    
    [string]$calibfile,

    [string]$trajectoryfile
)

$ErrorActionPreference = "Stop"

Write-Host "=== Insta360 Migration Pipeline ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Input Dir:        $inputdir"
Write-Host "INSV Path:        $insvpath"
Write-Host "MOV Path:         $movpath"
Write-Host "Calibration:      $calibfile"
Write-Host "Output Dir:       $outputdir"
Write-Host "Trajectory File:  $trajectoryfile"
Write-Host ""

# Resolve paths to absolute
$inputdir = (Resolve-Path $inputdir).Path
$insvpath = (Resolve-Path $insvpath).Path
$outputdir = if ([IO.Path]::IsPathRooted($outputdir)) { $outputdir } else { (Join-Path (Get-Location) $outputdir) }

# Create output directory
if (!(Test-Path $outputdir)) {
    Write-Host "Creating output directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $outputdir -Force | Out-Null
}

# Verify inputs exist
if (!(Test-Path $inputdir)) {
    Write-Host "ERROR: Input directory not found: $inputdir" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $insvpath)) {
    Write-Host "ERROR: INSV file not found: $insvpath" -ForegroundColor Red
    exit 1
}

# Find EXE files
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$BUILD_PACK = Join-Path $PROJECT_ROOT "build-pack"

$convert_manifold_exe = Join-Path $BUILD_PACK "convert_manifold.exe"
$insta_extraction_exe = Join-Path $BUILD_PACK "insta_data_extraction_ar.exe"
$insta_sync_exe = Join-Path $BUILD_PACK "insta_time_sync.exe"
$insta_poses_exe = Join-Path $BUILD_PACK "insta_compute_poses.exe"

# Check if EXEs exist
if (!(Test-Path $insta_extraction_exe)) {
    Write-Host "ERROR: insta_data_extraction.exe not found in $BUILD_PACK" -ForegroundColor Red
    Write-Host "Please run: .\ztools\build-pack.ps1" -ForegroundColor Yellow
    exit 1
}

# Step 1: Convert Manifold Livox lidar and IMU data
Write-Host "=== Step 1: Converting Manifold Livox/IMU data ===" -ForegroundColor Cyan

if (!(Test-Path $convert_manifold_exe)) {
    Write-Host "ERROR: convert_manifold.exe not found at: $convert_manifold_exe" -ForegroundColor Red
    exit 1
}

if (-not ($calibfile -and (Test-Path $calibfile))) {
    Write-Host "ERROR: Calibration file not found or not provided: $calibfile" -ForegroundColor Red
    Write-Host "  Usage: -calibfile path/to/calibration.dat" -ForegroundColor Yellow
    exit 1
}

Write-Host "Running: $convert_manifold_exe" -ForegroundColor Yellow
Write-Host "  Input Dir: $inputdir" -ForegroundColor Gray
Write-Host "  Calib: $calibfile" -ForegroundColor Gray

& $convert_manifold_exe `
    --project_dir $inputdir `
    --output_dir $outputdir

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Manifold conversion failed" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Manifold conversion completed" -ForegroundColor Green

# Step 2: Extract camera data from INSV
Write-Host "`n=== Step 2: Extracting camera data ===" -ForegroundColor Cyan
$camera_dir = Join-Path $outputdir "images"

& $insta_extraction_exe `
    --input-video-filename $insvpath `
    --output-dir $camera_dir `
    --no-export-frames

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data extraction failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Camera data extracted to: $camera_dir" -ForegroundColor Green


# Step 3: Run time synchronization and pose computation tools
Write-Host "`n=== Step 3: Synchronizing IMU data ===" -ForegroundColor Cyan

# Check if time sync tool exists
if (!(Test-Path $insta_sync_exe)) {
    Write-Host "ERROR: insta_time_sync.exe not found at $insta_sync_exe" -ForegroundColor Red
    exit 1
}

$imu_insv_dat = Join-Path $camera_dir "insv.dat"
$device_imu_dat = Join-Path $outputdir "imu.dat"

# Check if required input files exist
if (!(Test-Path $device_imu_dat)) {
    Write-Host "ERROR: Device IMU file not found: $device_imu_dat" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $imu_insv_dat)) {
    Write-Host "ERROR: Insta IMU file not found: $imu_insv_dat" -ForegroundColor Red
    exit 1
}

Write-Host "Running IMU time synchronization..." -ForegroundColor Yellow

$sync_cmd = "$insta_sync_exe --device $device_imu_dat --insta $imu_insv_dat"
Write-Host "Running: $sync_cmd" -ForegroundColor Gray

# Capture output to extract time delay
$stdout_file = [System.IO.Path]::GetTempFileName()
$stderr_file = [System.IO.Path]::GetTempFileName()

try {
    $sync_proc = Start-Process `
        -FilePath $insta_sync_exe `
        -ArgumentList @("--device", $device_imu_dat, "--insta", $imu_insv_dat) `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdout_file `
        -RedirectStandardError $stderr_file

    $sync_exit_code = $sync_proc.ExitCode
    $stdout_text = if (Test-Path $stdout_file) { Get-Content -Path $stdout_file -Raw } else { "" }
    $stderr_text = if (Test-Path $stderr_file) { Get-Content -Path $stderr_file -Raw } else { "" }
    $output_text = ($stdout_text + "`n" + $stderr_text).Trim()
}
finally {
    Remove-Item -Path $stdout_file -ErrorAction SilentlyContinue
    Remove-Item -Path $stderr_file -ErrorAction SilentlyContinue
}

# Convert output to string for regex matching
if ($output_text) {
    Write-Host $output_text
}

if ($sync_exit_code -ne 0) {
    Write-Host "ERROR: insta_time_sync failed with exit code $sync_exit_code" -ForegroundColor Red
    exit $sync_exit_code
}

# Extract time delay from output: "final time delay (s): NUMBER"
if ($output_text -match 'final time delay \(s\):\s*([\d\-\.]+)') {
    $time_offset = [double]$matches[1]
    Write-Host "[OK] Extracted time delay: $time_offset seconds" -ForegroundColor Green
} else {
    Write-Host "ERROR: Could not extract time delay from insta_time_sync output" -ForegroundColor Red
    Write-Host "Expected to find: 'final time delay (s): <number>'" -ForegroundColor Red
    exit 1
}

# Step 4: Extract images with timestamp offsets (using extracted time delay)
& $insta_extraction_exe `
    --input-video-filename $insvpath `
    --input-cam2-video-filename $movpath `
    --output-dir $camera_dir `
    --time-offset $time_offset `
    --export-frames

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data extraction failed" -ForegroundColor Red
    exit 1
}

# remove images/cam0 and images/cam1 if they exist (from first extraction without time offset), keep only images/cam2
Get-ChildItem -Path $camera_dir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^cam[01]$' } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Step 5: Compute camera poses
Write-Host "`n=== Step 5: Computing camera poses ===" -ForegroundColor Cyan

if (Test-Path $insta_poses_exe) {
    # Look for pose file in laser mapping output directory
    $traj_file = $trajectoryfile
    $calib_file = $calibfile
    
    if ((Test-Path $traj_file) -and (Test-Path $calib_file)) {
        $img_pose_output = Join-Path $camera_dir "ImgPose.txt"
        
        Write-Host "Using trajectory: $traj_file" -ForegroundColor Gray
        Write-Host "Using calibration: $calib_file" -ForegroundColor Gray
        
        & $insta_poses_exe `
            --poses-file $traj_file `
            --calib-file $calib_file `
            --image-folder $camera_dir `
            --output $img_pose_output
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Camera poses computed: $img_pose_output" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Pose computation failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARN] Trajectory or calibration file not found" -ForegroundColor Yellow
        Write-Host "  Expected in: $outputdir" -ForegroundColor Yellow
        Write-Host "  Looking for: trajectory_opt.txt and calibration.dat" -ForegroundColor Yellow
        Write-Host "  Hint: LaserMapping should have generated these files" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] insta_compute_poses.exe not found, skipping pose computation" -ForegroundColor Yellow
}

Write-Host "`n=== Pipeline Complete ===" -ForegroundColor Green
Write-Host "Output directory: $outputdir" -ForegroundColor Cyan

