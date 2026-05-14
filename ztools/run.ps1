#!/usr/bin/env pwsh
# Run Insta360 Migration Pipeline
# Usage: .\run.ps1 -inputdir <input/dir> -insvpath <video.insv> -outputdir <output/dir> [-calibfile <calibration.dat>]

param(
    [Parameter(Mandatory=$true)]
    [string]$inputdir,
    
    [Parameter(Mandatory=$true)]
    [string]$insvpath,
    
    [Parameter(Mandatory=$true)]
    [string]$outputdir,
    
    [string]$calibfile = ""
)

$ErrorActionPreference = "Stop"

Write-Host "=== Insta360 Migration Pipeline ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Input Dir:        $inputdir"
Write-Host "INSV Path:        $insvpath"
Write-Host "Calibration:      $calibfile"
Write-Host "Output Dir:       $outputdir"
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

$output_calibfile = Join-Path $outputdir "calibration.dat"

if ($calibfile) {
    if (!(Test-Path $calibfile)) {
        Write-Host "ERROR: Calibration file not found: $calibfile" -ForegroundColor Red
        exit 1
    }

    Write-Host "Copying calibration.dat to output directory..." -ForegroundColor Yellow
    Copy-Item -Path $calibfile -Destination $output_calibfile -Force
    $calibfile = $output_calibfile
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

# All executables are in build-pack directory
$NATIVE_TOOLS = $BUILD_PACK
$PYTHON_TOOLS = $BUILD_PACK

$convert_manifold_exe = Join-Path $BUILD_PACK "convert_manifold.exe"
$insta_extraction_exe = Join-Path $PYTHON_TOOLS "insta_data_extraction.exe"
$insta_poses_exe = Join-Path $PYTHON_TOOLS "insta_compute_poses.exe"
$lasermapping_exe = Join-Path $NATIVE_TOOLS "slam.exe"
$slam_post_exe = Join-Path $NATIVE_TOOLS "slam_post.exe"

# Check if EXEs exist
if (!(Test-Path $insta_extraction_exe)) {
    Write-Host "ERROR: insta_data_extraction.exe not found in $BUILD_PACK" -ForegroundColor Red
    Write-Host "Please run: .\ztools\build-pack.ps1" -ForegroundColor Yellow
    exit 1
}

# Step 1: Convert Manifold Livox lidar and IMU data
Write-Host "=== Step 1: Converting Manifold Livox/IMU data ===" -ForegroundColor Cyan
    
    if (Test-Path $convert_manifold_exe) {
        if ($calibfile -and (Test-Path $calibfile)) {
            Write-Host "Running: $convert_manifold_exe" -ForegroundColor Yellow
            Write-Host "  Input Dir: $inputdir" -ForegroundColor Gray
            Write-Host "  Calib: $calibfile" -ForegroundColor Gray
            
            & $convert_manifold_exe `
                --project_dir $inputdir `
                --output_dir $outputdir
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Manifold conversion completed" -ForegroundColor Green
            } else {
                Write-Host "[WARN] Manifold conversion failed" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[WARN] Calibration file not found or not provided: $calibfile" -ForegroundColor Yellow
            Write-Host "  Usage: -calibfile path/to/calibration.dat" -ForegroundColor Yellow
        }
} else {
    Write-Host "[WARN] convert_manifold.exe not found at: $convert_manifold_exe" -ForegroundColor Yellow
}

# Step 2: Extract, synchronize, and export camera data from INSV
Write-Host "`n=== Step 2: Extracting and synchronizing camera data ===" -ForegroundColor Cyan
$camera_dir = Join-Path $outputdir "images"
$device_imu_dat = Join-Path $outputdir "imu.dat"

if (!(Test-Path $device_imu_dat)) {
    Write-Host "ERROR: Device IMU file not found: $device_imu_dat" -ForegroundColor Red
    exit 1
}

& $insta_extraction_exe `
    --input-video-filename $insvpath `
    --output-dir $camera_dir `
    --imu-file $device_imu_dat

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data extraction failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Camera data extracted to: $camera_dir" -ForegroundColor Green

# Step 3: Run lasermapping
Write-Host "`n=== Step 3: Computing trajectory from lidar data ===" -ForegroundColor Yellow
    Write-Host "[WARN] This step requires pre-processed lidar data from Manifold conversion" -ForegroundColor Yellow
    Write-Host "  Note: slam needs pre-processed .dat files in $outputdir" -ForegroundColor Yellow
    Write-Host "  - calibration.dat" -ForegroundColor Gray
    Write-Host "  - imu.dat" -ForegroundColor Gray
    Write-Host "  - encoder.dat" -ForegroundColor Gray
    Write-Host "  - lidar.dat" -ForegroundColor Gray
    
    # If these files already exist, slam can process them
    $has_dat_files = (Test-Path (Join-Path $outputdir "imu.dat")) -and `
                     (Test-Path (Join-Path $outputdir "calibration.dat"))
    
    if ($has_dat_files) {
        Write-Host "  Found data files, attempting to compute trajectory..." -ForegroundColor Green
        if (Test-Path $lasermapping_exe) {
            Write-Host "  Running: $lasermapping_exe --project_dir $outputdir" -ForegroundColor Gray
            
            # Try with = format first (gflags standard)
            & $lasermapping_exe "-project_dirname=$outputdir" "-output_dir=$outputdir"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Trajectory computed" -ForegroundColor Green
            } else {
                Write-Host "[WARN] Trajectory computation failed (may need ROS environment)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[WARN] slam.exe not found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Data files not found, skipping trajectory computation" -ForegroundColor Yellow
        Write-Host "  To process: Run Manifold conversion first or extract .dat files manually" -ForegroundColor Yellow
    }

# Step 4: Run slam_post.exe to refine trajectory (if slam.exe succeeded)
Write-Host "`n=== Step 4: Refining trajectory with slam_post.exe ===" -ForegroundColor Cyan
if (Test-Path $slam_post_exe) {
    $pgo_config_file = Join-Path $BUILD_PACK "pgo.json"
    if (-not (Test-Path $pgo_config_file)) {
        Write-Host "ERROR: pgo.json not found in build-pack. Please run .\ztools\build-pack.ps1" -ForegroundColor Red
        exit 1
    }

    Write-Host "Running: $slam_post_exe --config_filename $pgo_config_file --project_input_path $outputdir --project_output_path $outputdir" -ForegroundColor Gray
    
    & $slam_post_exe "--config_filename=$pgo_config_file" "--project_input_path=$outputdir" "--project_output_path=$outputdir"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Trajectory refined with slam_post.exe" -ForegroundColor Green
    } else {
        Write-Host "[WARN] slam_post.exe failed to refine trajectory" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] slam_post.exe not found, skipping trajectory refinement" -ForegroundColor Yellow
}

# Step 5: Compute camera poses
Write-Host "`n=== Step 5: Computing camera poses ===" -ForegroundColor Cyan

if (Test-Path $insta_poses_exe) {
    # Look for pose file in laser mapping output directory
    $traj_file = Join-Path $outputdir "trajectory_opt.txt"
    $calib_file = Join-Path $outputdir "calibration.dat"
    
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

# Summary
Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "Pipeline Complete" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "`nOutput directory: $outputdir" -ForegroundColor Cyan

# Show file tree
if (Test-Path $outputdir) {
    Write-Host "`nOutput files:" -ForegroundColor Yellow
    Get-ChildItem -Path $outputdir -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        $indent = ("  " * ($_.FullName -replace "[^\\]" | Measure-Object -Character).Characters)
        if ($_.PSIsContainer) {
            Write-Host "$indent[DIR] $($_.Name)/"
        } else {
            $size = if ($_.Length -gt 1MB) { "$([Math]::Round($_.Length/1MB, 2)) MB" } else { "$([Math]::Round($_.Length/1KB, 2)) KB" }
            Write-Host "$indent[FILE] $($_.Name) ($size)"
        }
    }
}

