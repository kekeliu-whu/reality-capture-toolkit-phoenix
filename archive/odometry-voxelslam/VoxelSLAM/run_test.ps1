# VoxelSLAM Windows test runner
# Usage: .\run_test.ps1 [-Config <path>] [-Bag <path>] [-Output <dir>]

param(
  [string]$Config = "config/bag_test.yaml",
  [string]$Bag = "D:/tmp/bag_20260525_163839.bag",
  [string]$Output = "D:/tmp/voxelslam_output",
  [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $Output | Out-Null
New-Item -ItemType Directory -Force -Path "$Output/bag_20260525_163839" | Out-Null

# Set up DLL search path (migration DLLs)
$migrationDlls = "$PSScriptRoot\..\..\migration\build\RelWithDebInfo"
$voxelDlls = "$PSScriptRoot\..\..\migration\build\voxelslam\RelWithDebInfo"
$env:PATH = "$migrationDlls;$voxelDlls;$env:PATH"

# Run voxelslam
$exe = "$PSScriptRoot\$BuildDir\RelWithDebInfo\voxelslam.exe"
if (-not (Test-Path $exe)) {
  $exe = "$PSScriptRoot\$BuildDir\Release\voxelslam.exe"
}
if (-not (Test-Path $exe)) {
  Write-Error "voxelslam.exe not found in $BuildDir. Build first."
  exit 1
}

Write-Host "Running: $exe --bag $Bag --config $Config"
Write-Host "Output dir: $Output"
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$startTime = Get-Date
& $exe --bag $Bag --config $Config 2>&1 | Tee-Object -FilePath "$Output/run.log"
$exitCode = $LASTEXITCODE
$elapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "Exit code: $exitCode"
Write-Host "Elapsed: $($elapsed.TotalMinutes.ToString('F1')) minutes"
Write-Host "Output: $Output/bag_20260525_163839/"
exit $exitCode
