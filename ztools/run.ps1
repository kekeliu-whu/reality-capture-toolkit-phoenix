#!/usr/bin/env pwsh
# Run Insta360 Migration Pipeline
# Usage: .\.run.ps1 -bagpath <imu.dat> -insvpath <video.insv> -outputdir <output/dir> [-calibfile <calib.yaml>] [-skip-convert_s20] [-skip-lasermapping]

param(
    [Parameter(Mandatory=$true)]
    [string]$bagpath,
    
    [Parameter(Mandatory=$true)]
    [string]$insvpath,
    
    [Parameter(Mandatory=$true)]
    [string]$outputdir,
    
    [string]$calibfile = "",
    
    [switch]$skip_convert_s20,
    
    [switch]$skip_lasermapping
)

$ErrorActionPreference = "Stop"

Write-Host "=== Insta360 Migration Pipeline ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bag Path:         $bagpath"
Write-Host "INSV Path:        $insvpath"
Write-Host "Calibration:      $calibfile"
Write-Host "Output Dir:       $outputdir"
Write-Host "Skip Convert-S20: $skip_convert_s20"
Write-Host "Skip LaserMapping: $skip_lasermapping"
Write-Host ""

# Resolve paths to absolute
$bagpath = (Resolve-Path $bagpath).Path
$insvpath = (Resolve-Path $insvpath).Path
$outputdir = if ([IO.Path]::IsPathRooted($outputdir)) { $outputdir } else { (Join-Path (Get-Location) $outputdir) }

# Create output directory
if (!(Test-Path $outputdir)) {
    Write-Host "Creating output directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $outputdir -Force | Out-Null
}

# Verify inputs exist
if (!(Test-Path $bagpath)) {
    Write-Host "ERROR: Bag file not found: $bagpath" -ForegroundColor Red
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

$convert_s20_exe = Join-Path $BUILD_PACK "convert_s20.exe"
$insta_extraction_exe = Join-Path $PYTHON_TOOLS "insta_data_extraction.exe"
$insta_sync_exe = Join-Path $PYTHON_TOOLS "insta_time_sync.exe"
$insta_poses_exe = Join-Path $PYTHON_TOOLS "insta_compute_poses.exe"
$lasermapping_exe = Join-Path $NATIVE_TOOLS "slam_core_main.exe"

# Check if EXEs exist
if (!(Test-Path $insta_extraction_exe)) {
    Write-Host "ERROR: insta_data_extraction.exe not found in $BUILD_PACK" -ForegroundColor Red
    Write-Host "Please run: .\ztools\build-pack.ps1" -ForegroundColor Yellow
    exit 1
}

# Step 1: Convert S20 Livox lidar and IMU data (optional)
if (!$skip_convert_s20) {
    Write-Host "=== Step 1: Converting S20 Livox/IMU data ===" -ForegroundColor Cyan
    
    if (Test-Path $convert_s20_exe) {
        if ($calibfile -and (Test-Path $calibfile)) {
            Write-Host "Running: $convert_s20_exe" -ForegroundColor Yellow
            Write-Host "  Bag: $bagpath" -ForegroundColor Gray
            Write-Host "  Calib: $calibfile" -ForegroundColor Gray
            
            & $convert_s20_exe `
                --bag_filename $bagpath `
                --calib_filename $calibfile `
                --output_dir $outputdir
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ S20 conversion completed" -ForegroundColor Green
            } else {
                Write-Host "⚠ S20 conversion failed" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠ Calibration file not found or not provided: $calibfile" -ForegroundColor Yellow
            Write-Host "  Usage: -calibfile path/to/calib.yaml" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ convert_s20.exe not found at: $convert_s20_exe" -ForegroundColor Yellow
    }
} else {
    Write-Host "=== Step 1: Skipping S20 conversion ===" -ForegroundColor Yellow
}

# Step 2: Extract camera data from INSV
Write-Host "`n=== Step 2: Extracting camera data ===" -ForegroundColor Cyan
$camera_dir = Join-Path $outputdir "camera"

& $insta_extraction_exe `
    --input-video-filename $insvpath `
    --output-dir $camera_dir

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data extraction failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Camera data extracted to: $camera_dir" -ForegroundColor Green

# Step 3: Run lasermapping if not skipped
if (!$skip_lasermapping) {
    Write-Host "`n=== Step 3: Computing trajectory from lidar data ===" -ForegroundColor Yellow
    Write-Host "⚠ This step requires pre-processed lidar data from S20 conversion" -ForegroundColor Yellow
    Write-Host "  Note: slam_core_main needs pre-processed .dat files in $outputdir" -ForegroundColor Yellow
    Write-Host "  - calibration.dat" -ForegroundColor Gray
    Write-Host "  - imu.dat" -ForegroundColor Gray
    Write-Host "  - encoder.dat" -ForegroundColor Gray
    Write-Host "  - lidar.dat" -ForegroundColor Gray
    
    # If these files already exist, slam_core_main can process them
    $has_dat_files = (Test-Path (Join-Path $outputdir "imu.dat")) -and `
                     (Test-Path (Join-Path $outputdir "calibration.dat"))
    
    if ($has_dat_files) {
        Write-Host "  Found data files, attempting to compute trajectory..." -ForegroundColor Green
        if (Test-Path $lasermapping_exe) {
            Write-Host "  Running: $lasermapping_exe --project_dir $outputdir" -ForegroundColor Gray
            
            # Try with = format first (gflags standard)
            & $lasermapping_exe "-project_dirname=$outputdir" "-output_dir=$outputdir"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ Trajectory computed" -ForegroundColor Green
            } else {
                Write-Host "⚠ Trajectory computation failed (may need ROS environment)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠ slam_core_main.exe not found" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Data files not found, skipping trajectory computation" -ForegroundColor Yellow
        Write-Host "  To process: Run S20 conversion first or extract .dat files manually" -ForegroundColor Yellow
    }
}

# Step 4: Synchronize IMU data
Write-Host "`n=== Step 4: Synchronizing IMU data ===" -ForegroundColor Cyan

# Check if time sync is needed
if (Test-Path $insta_sync_exe) {
    Write-Host "Attempting IMU time synchronization..." -ForegroundColor Yellow
    
    $imu_insv_dat = Join-Path $camera_dir "insv.dat"
    $device_imu_dat = Join-Path $outputdir "imu.dat"
    
    if (Test-Path $device_imu_dat) {
        & $insta_sync_exe `
            --device $device_imu_dat `
            --insta $imu_insv_dat
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠ IMU sync failed, continuing without sync" -ForegroundColor Yellow
        } else {
            Write-Host "✓ IMU time synchronized" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠ Device IMU file not found: $device_imu_dat" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ insta_time_sync.exe not found, skipping IMU synchronization" -ForegroundColor Yellow
}

# Step 5: Compute camera poses
Write-Host "`n=== Step 5: Computing camera poses ===" -ForegroundColor Cyan

if (Test-Path $insta_poses_exe) {
    # Look for pose file in laser mapping output directory
    $traj_file = Join-Path $outputdir "traj.txt"
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
            Write-Host "✓ Camera poses computed: $img_pose_output" -ForegroundColor Green
        } else {
            Write-Host "⚠ Pose computation failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ Trajectory or calibration file not found" -ForegroundColor Yellow
        Write-Host "  Expected in: $outputdir" -ForegroundColor Yellow
        Write-Host "  Looking for: traj.txt and calibration.dat" -ForegroundColor Yellow
        Write-Host "  Hint: LaserMapping should have generated these files" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ insta_compute_poses.exe not found, skipping pose computation" -ForegroundColor Yellow
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
            Write-Host "$indent📁 $($_.Name)/"
        } else {
            $size = if ($_.Length -gt 1MB) { "$([Math]::Round($_.Length/1MB, 2)) MB" } else { "$([Math]::Round($_.Length/1KB, 2)) KB" }
            Write-Host "$indent📄 $($_.Name) ($size)"
        }
    }
}

