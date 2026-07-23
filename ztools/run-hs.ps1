#!/usr/bin/env pwsh

<#
.SYNOPSIS
Fully process a Kosmo/Hesai (HS) Raw Data directory.

.DESCRIPTION
Runs the HS MCAP converter, exports transformed camera images, executes
REALTIME_MAPPING, optimizes the body/IMU trajectory with slam_post, and writes
per-image camera poses to images/ImgPose.txt.

.EXAMPLE
.\ztools\run-hs.ps1 -InputDir "D:\data_20260719_133820\Raw Data"

.EXAMPLE
.\ztools\run-hs.ps1 `
  -InputDir "D:\data_20260719_133820\Raw Data" `
  -OutputDir "D:\data_20260719_133820"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [string]$OutputDir = "",
    [string]$PgoConfig = "",
    [string]$FfmpegPath = "",
    [string]$ConvertHsExe = "",
    [string]$RealtimeMappingExe = "",
    [string]$RealtimeConfig = "",
    [string]$SlamPostExe = "",
    [string]$ComputePosesExe = "",

    [switch]$SkipConversion,
    [switch]$SkipSlam,
    [switch]$SkipPgo,
    [switch]$SkipPoses,
    [switch]$KeepExistingImages
)

$ErrorActionPreference = "Stop"
$script:StepNumber = 0

function Write-Step {
    param([string]$Text)
    $script:StepNumber++
    Write-Host ""
    Write-Host "=== Step $($script:StepNumber): $Text ===" -ForegroundColor Cyan
}

function Resolve-ExistingFile {
    param(
        [string]$Description,
        [string]$Override,
        [string[]]$Candidates
    )

    if ($Override) {
        if (-not (Test-Path -LiteralPath $Override -PathType Leaf)) {
            throw "$Description not found: $Override"
        }
        return (Resolve-Path -LiteralPath $Override).Path
    }

    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "$Description not found. Checked:`n  $($Candidates -join "`n  ")"
}

function Resolve-BuildPackFile {
    param(
        [string]$Description,
        [string]$Override,
        [string]$BuildPackDir,
        [string]$FileName
    )

    $resolvedBuildPackDir = [IO.Path]::GetFullPath($BuildPackDir).TrimEnd('\')
    $candidate = if ($Override) { $Override } else { Join-Path $BuildPackDir $FileName }
    $resolved = Resolve-ExistingFile -Description $Description -Override $candidate -Candidates @()
    $resolvedFullPath = [IO.Path]::GetFullPath($resolved)

    if (-not $resolvedFullPath.StartsWith($resolvedBuildPackDir + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must come from build-pack: $resolvedFullPath"
    }

    return $resolvedFullPath
}

function Invoke-NativeStep {
    param(
        [string]$Description,
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host "Running: $Executable" -ForegroundColor Gray
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
    Write-Host "[OK] $Description" -ForegroundColor Green
}

function Remove-GeneratedFiles {
    param(
        [string]$Directory,
        [string[]]$Names
    )

    $resolvedDirectory = [IO.Path]::GetFullPath($Directory)
    foreach ($name in $Names) {
        $target = [IO.Path]::GetFullPath((Join-Path $resolvedDirectory $name))
        if (-not $target.StartsWith($resolvedDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove a generated file outside the output directory: $target"
        }
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            Remove-Item -LiteralPath $target -Force
        }
    }
}

function Assert-File {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not generated: $Path"
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$BuildPack = Join-Path $ProjectRoot "build-pack"

$inputCandidate = [IO.Path]::GetFullPath($InputDir)
if (-not (Test-Path -LiteralPath $inputCandidate -PathType Container)) {
    throw "Input directory not found: $inputCandidate"
}

if (Test-Path -LiteralPath (Join-Path $inputCandidate "calib_info.yaml") -PathType Leaf) {
    $RawDataDir = (Resolve-Path -LiteralPath $inputCandidate).Path
} elseif (Test-Path -LiteralPath (Join-Path $inputCandidate "Raw Data\calib_info.yaml") -PathType Leaf) {
    $RawDataDir = (Resolve-Path -LiteralPath (Join-Path $inputCandidate "Raw Data")).Path
} else {
    throw "Input must be an HS Raw Data directory (or its parent) containing calib_info.yaml"
}

if (-not $OutputDir) {
    if ((Split-Path -Leaf $RawDataDir) -eq "Raw Data") {
        $OutputDir = Split-Path -Parent $RawDataDir
    } else {
        $OutputDir = Join-Path (Split-Path -Parent $RawDataDir) "output"
    }
}
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if ($OutputDir.TrimEnd('\') -eq $RawDataDir.TrimEnd('\')) {
    throw "OutputDir must not be the Raw Data input directory"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ConvertHsExe = Resolve-BuildPackFile -Description "convert_hs.exe" -Override $ConvertHsExe -BuildPackDir $BuildPack -FileName "convert_hs.exe"
$RealtimeMappingExe = Resolve-BuildPackFile -Description "REALTIME_MAPPING.exe" -Override $RealtimeMappingExe -BuildPackDir $BuildPack -FileName "REALTIME_MAPPING.exe"
$RealtimeConfig = Resolve-BuildPackFile -Description "L2PRO.yaml" -Override $RealtimeConfig -BuildPackDir $BuildPack -FileName "L2PRO.yaml"
$SlamPostExe = Resolve-BuildPackFile -Description "slam_post.exe" -Override $SlamPostExe -BuildPackDir $BuildPack -FileName "slam_post.exe"
$ComputePosesExe = Resolve-BuildPackFile -Description "insta_compute_poses.exe" -Override $ComputePosesExe -BuildPackDir $BuildPack -FileName "insta_compute_poses.exe"
$PgoConfig = Resolve-BuildPackFile -Description "pgo.json" -Override $PgoConfig -BuildPackDir $BuildPack -FileName "pgo.json"

$FfmpegPath = Resolve-BuildPackFile -Description "ffmpeg.exe" -Override $FfmpegPath -BuildPackDir $BuildPack -FileName "ffmpeg.exe"

$runtimeDirs = @(
    (Split-Path -Parent $ConvertHsExe),
    (Split-Path -Parent $RealtimeMappingExe),
    (Split-Path -Parent $SlamPostExe)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } | Select-Object -Unique
$env:PATH = (($runtimeDirs + @($env:PATH)) -join [IO.Path]::PathSeparator)
$env:PYTHONUTF8 = "1"

Write-Host "=== HS automatic processing pipeline ===" -ForegroundColor Cyan
Write-Host "Raw Data:          $RawDataDir"
Write-Host "Output:            $OutputDir"
Write-Host "convert_hs:        $ConvertHsExe"
Write-Host "realtime mapping:  $RealtimeMappingExe"
Write-Host "realtime config:   $RealtimeConfig"
Write-Host "PGO:               $SlamPostExe"
Write-Host "Image pose export: $ComputePosesExe"

if (-not $SkipConversion) {
    Write-Step "Convert HS MCAP data and export images"
    $convertArgs = @(
        "--input_dir=$RawDataDir",
        "--output_dir=$OutputDir",
        "--ffmpeg_path=$FfmpegPath",
        "--image_flip_horizontal=true",
        "--image_rotate_cw_180=true",
        "--clear_image_camera_dirs=$((-not $KeepExistingImages).ToString().ToLowerInvariant())"
    )
    Invoke-NativeStep -Description "HS conversion" -Executable $ConvertHsExe -Arguments $convertArgs
}

$CalibrationFile = Join-Path $OutputDir "calibration.dat"
$ImuFile = Join-Path $OutputDir "imu.dat"
$LidarFile = Join-Path $OutputDir "lidar.dat"
$ImagesDir = Join-Path $OutputDir "images"
Assert-File $CalibrationFile "Calibration"
Assert-File $ImuFile "IMU data"
Assert-File $LidarFile "LiDAR data"
foreach ($cameraName in @("cam0", "cam1", "cam2")) {
    $cameraDir = Join-Path $ImagesDir $cameraName
    $imageCount = @(Get-ChildItem -LiteralPath $cameraDir -Filter "*.jpg" -File -ErrorAction SilentlyContinue).Count
    if ($imageCount -eq 0) {
        throw "No exported images found in $cameraDir"
    }
    Write-Host "  $cameraName images: $imageCount" -ForegroundColor Gray
}

if (-not $SkipSlam) {
    Write-Step "Run realtime body/IMU SLAM"
    Remove-GeneratedFiles -Directory $OutputDir -Names @("trajectory.txt", "map.las", "traj.dat", "lidar_undist.dat")
    Invoke-NativeStep -Description "Realtime SLAM" -Executable $RealtimeMappingExe -Arguments @(
        "--input_dir=$OutputDir",
        "--output_dir=$OutputDir",
        "--config_filename=$RealtimeConfig"
    )
}

$TrajectoryFile = Join-Path $OutputDir "trajectory.txt"
$TrajectoryDat = Join-Path $OutputDir "traj.dat"
$UndistortedLidar = Join-Path $OutputDir "lidar_undist.dat"
Assert-File $TrajectoryFile "Realtime trajectory"
Assert-File $TrajectoryDat "Realtime body trajectory"
Assert-File $UndistortedLidar "Realtime body-frame undistorted LiDAR"

if (-not $SkipPgo) {
    Write-Step "Optimize body/IMU trajectory with PGO"
    Remove-GeneratedFiles -Directory $OutputDir -Names @("trajectory_opt.txt", "map_opt.las", "pgo_metrics.dat")
    Invoke-NativeStep -Description "PGO" -Executable $SlamPostExe -Arguments @(
        "--config_filename=$PgoConfig",
        "--project_input_path=$OutputDir",
        "--project_output_path=$OutputDir"
    )
}

$OptimizedTrajectory = Join-Path $OutputDir "trajectory_opt.txt"
$OptimizedMap = Join-Path $OutputDir "map_opt.las"
Assert-File $OptimizedTrajectory "Optimized body/IMU trajectory"
Assert-File $OptimizedMap "Optimized point cloud"

if (-not $SkipPoses) {
    Write-Step "Compute per-image camera poses"
    $ImgPoseFile = Join-Path $ImagesDir "ImgPose.txt"
    if (Test-Path -LiteralPath $ImgPoseFile -PathType Leaf) {
        Remove-Item -LiteralPath $ImgPoseFile -Force
    }
    Invoke-NativeStep -Description "Image pose export" -Executable $ComputePosesExe -Arguments @(
        "--poses-file", $OptimizedTrajectory,
        "--calib-file", $CalibrationFile,
        "--image-folder", $ImagesDir,
        "--output", $ImgPoseFile
    )
}

$ImgPoseFile = Join-Path $ImagesDir "ImgPose.txt"
Assert-File $ImgPoseFile "ImgPose.txt"
$poseRows = [Math]::Max(0, @(Get-Content -LiteralPath $ImgPoseFile).Count - 1)

Write-Host ""
Write-Host "=== HS pipeline complete ===" -ForegroundColor Green
Write-Host "Output:               $OutputDir"
Write-Host "Optimized trajectory: $OptimizedTrajectory"
Write-Host "Optimized map:        $OptimizedMap"
Write-Host "Image poses:          $ImgPoseFile ($poseRows rows)"
