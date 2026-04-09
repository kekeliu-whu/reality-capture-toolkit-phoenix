#!/usr/bin/env pwsh
# Comprehensive build script for XColor, Odometry-Old, and Python Tools
# Builds C++ projects and compiles Python tools to EXE
# Usage: .\build-all.ps1 [-TestOnly]

param(
    [switch]$TestOnly = $false
)

# Activate Python virtual environment
Write-Host "Activating Python virtual environment..." -ForegroundColor Yellow
& d:\ProjectX\project-3d\reality-capture-toolkit\.venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "Virtual environment activated successfully" -ForegroundColor Green
Write-Host ""

# CMake executable path
$CMAKE_EXE = "D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$VCPKG_TOOLCHAIN_FILE = "F:/Library/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Project paths
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$BUILD_ALL_DIR = Join-Path $PROJECT_ROOT "build-all"
$XCOLOR_SOURCE_DIR = Join-Path $PROJECT_ROOT "xcolor"
$XCOLOR_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-xcolor"
$ODOMETRY_OLD_SOURCE_DIR = Join-Path $PROJECT_ROOT "odometry-old"
$ODOMETRY_OLD_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-odometry-old"
$PGO_SOURCE_DIR = Join-Path $PROJECT_ROOT "pgo"
$PGO_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-pgo"
$SCRIPTS_DIR = Join-Path $PROJECT_ROOT "migration\scripts"
$PYTHON_TOOLS_DIR = Join-Path $BUILD_ALL_DIR "build-python-tools"

# ============================================================
# Title and Prerequisites Check
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Complete Build System" -ForegroundColor Cyan
Write-Host "  * XColor" -ForegroundColor Cyan
Write-Host "  * Odometry-Old" -ForegroundColor Cyan
Write-Host "  * Python Tools" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking prerequisites..." -ForegroundColor Yellow
Write-Host ""

# Check CMake executable
if (!(Test-Path $CMAKE_EXE)) {
    Write-Host "ERROR: CMake not found at $CMAKE_EXE" -ForegroundColor Red
    exit 1
}

# Check source directories
if (!(Test-Path $XCOLOR_SOURCE_DIR)) {
    Write-Host "ERROR: XColor source not found at $XCOLOR_SOURCE_DIR" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $ODOMETRY_OLD_SOURCE_DIR)) {
    Write-Host "ERROR: Odometry-Old source not found at $ODOMETRY_OLD_SOURCE_DIR" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $SCRIPTS_DIR)) {
    Write-Host "ERROR: Python scripts directory not found at $SCRIPTS_DIR" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $PGO_SOURCE_DIR)) {
    Write-Host "ERROR: PGO source not found at $PGO_SOURCE_DIR" -ForegroundColor Red
    exit 1
}

# Check Python files
$PYTHON_FILES = @(
    "insta_data_extraction.py",
    "insta_time_sync.py",
    "insta_compute_poses.py",
    "insta_compute_poses_ar.py"
)

$missing = $false
foreach ($file in $PYTHON_FILES) {
    $path = Join-Path $SCRIPTS_DIR $file
    if (!(Test-Path $path)) {
        Write-Host "ERROR: Python file not found - $file" -ForegroundColor Red
        $missing = $true
    }
}

if ($missing) {
    exit 1
}

Write-Host "[PASS] All prerequisites found" -ForegroundColor Green
Write-Host ""

# Clean up and create build-all directory
if (Test-Path $BUILD_ALL_DIR) {
    Write-Host "Cleaning up existing build-all directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $BUILD_ALL_DIR
}

Write-Host "Creating build-all directory..." -ForegroundColor Gray
New-Item -ItemType Directory -Path $BUILD_ALL_DIR -Force | Out-Null

Write-Host ""
Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Gray
Write-Host "XColor Build: $XCOLOR_BUILD_DIR" -ForegroundColor Gray
Write-Host "Odometry-Old Build: $ODOMETRY_OLD_BUILD_DIR" -ForegroundColor Gray
Write-Host "PGO Build: $PGO_BUILD_DIR" -ForegroundColor Gray
Write-Host "Python Tools: $PYTHON_TOOLS_DIR" -ForegroundColor Gray
Write-Host ""

if ($TestOnly) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "TEST MODE: Prerequisite check completed" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Run without -TestOnly flag to execute full build:" -ForegroundColor Yellow
    Write-Host "  .\build-all.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

# ============================================================
# Stage 1: Build XColor
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage 1/3: Building XColor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $XCOLOR_BUILD_DIR) {
    Write-Host "Removing existing build directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $XCOLOR_BUILD_DIR
}

Write-Host "Creating build directory..." -ForegroundColor Gray
New-Item -ItemType Directory -Path $XCOLOR_BUILD_DIR -Force | Out-Null

Write-Host "Configuring with CMake..." -ForegroundColor Yellow

& $CMAKE_EXE `
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE `
    "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN_FILE" `
    --no-warn-unused-cli `
    -S $XCOLOR_SOURCE_DIR `
    -B $XCOLOR_BUILD_DIR `
    -G "Visual Studio 17 2022" `
    -T "host=x64" `
    -A "x64"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: XColor CMake configuration failed" -ForegroundColor Red
    exit 1
}

Write-Host "Building..." -ForegroundColor Yellow

& $CMAKE_EXE `
    --build $XCOLOR_BUILD_DIR `
    --config "Release" `
    --target "ALL_BUILD" `
    -j 24

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: XColor build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] XColor build completed" -ForegroundColor Green
Write-Host ""

# ============================================================
# Stage 2: Build Odometry-Old
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage 2/3: Building Odometry-Old" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $ODOMETRY_OLD_BUILD_DIR) {
    Write-Host "Removing existing build directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $ODOMETRY_OLD_BUILD_DIR
}

Write-Host "Creating build directory..." -ForegroundColor Gray
New-Item -ItemType Directory -Path $ODOMETRY_OLD_BUILD_DIR -Force | Out-Null

Write-Host "Configuring with CMake..." -ForegroundColor Yellow

& $CMAKE_EXE `
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE `
    "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN_FILE" `
    --no-warn-unused-cli `
    -S $ODOMETRY_OLD_SOURCE_DIR `
    -B $ODOMETRY_OLD_BUILD_DIR `
    -G "Visual Studio 17 2022" `
    -T "host=x64" `
    -A "x64"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Odometry-Old CMake configuration failed" -ForegroundColor Red
    exit 1
}

Write-Host "Building..." -ForegroundColor Yellow

& $CMAKE_EXE `
    --build $ODOMETRY_OLD_BUILD_DIR `
    --config "Release" `
    --target "ALL_BUILD" `
    -j 24

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Odometry-Old build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Odometry-Old build completed" -ForegroundColor Green
Write-Host ""

# ============================================================
# Stage 2.5: Build PGO
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage 2.5/3: Building PGO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $PGO_BUILD_DIR) {
    Write-Host "Removing existing build directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $PGO_BUILD_DIR
}

Write-Host "Creating build directory..." -ForegroundColor Gray
New-Item -ItemType Directory -Path $PGO_BUILD_DIR -Force | Out-Null

Write-Host "Configuring with CMake..." -ForegroundColor Yellow

& $CMAKE_EXE `
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE `
    "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN_FILE" `
    --no-warn-unused-cli `
    -S $PGO_SOURCE_DIR `
    -B $PGO_BUILD_DIR `
    -G "Visual Studio 17 2022" `
    -T "host=x64" `
    -A "x64"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PGO CMake configuration failed" -ForegroundColor Red
    exit 1
}

Write-Host "Building..." -ForegroundColor Yellow

& $CMAKE_EXE `
    --build $PGO_BUILD_DIR `
    --config "Release" `
    --target "ALL_BUILD" `
    -j 24

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PGO build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] PGO build completed" -ForegroundColor Green
Write-Host ""

# ============================================================
# Stage 3: Build Python Tools
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage 3/3: Compiling Python Tools to EXE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to scripts directory
Push-Location $SCRIPTS_DIR

# Create output directory
if (!(Test-Path $PYTHON_TOOLS_DIR)) {
    New-Item -ItemType Directory -Path $PYTHON_TOOLS_DIR | Out-Null
}

Write-Host "Compiling Python files..." -ForegroundColor Yellow
Write-Host ""

$success = 0

foreach ($file in $PYTHON_FILES) {
    $name = [IO.Path]::GetFileNameWithoutExtension($file)
    
    Write-Host "$file..." -ForegroundColor Yellow
    
    # Run pyinstaller
    & pyinstaller -F $file -y
    $exit_code = $LASTEXITCODE
    
    # Check if EXE was created
    $exe_path = Join-Path (Get-Location) "dist\$name.exe"
    
    if (Test-Path $exe_path) {
        # Move to python-tools
        $output_exe = Join-Path $PYTHON_TOOLS_DIR "$name.exe"
        Move-Item $exe_path $output_exe -Force
        
        $size = (Get-Item $output_exe).Length / 1MB
        Write-Host "  [OK] ($([Math]::Round($size, 2)) MB)" -ForegroundColor Green
        $success++
    } else {
        Write-Host "  [FAILED] (exit code: $exit_code)" -ForegroundColor Red
    }
}

# Cleanup
Write-Host ""
Write-Host "Cleaning up temporary files..." -ForegroundColor Gray
Remove-Item build, dist, *.spec -Recurse -Force -ErrorAction SilentlyContinue

# Restore location
Pop-Location

Write-Host "[OK] Python tools compilation completed" -ForegroundColor Green
Write-Host ""

# ============================================================
# Final Summary
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BUILD SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[OK] XColor" -ForegroundColor Green
Write-Host "  Location: $XCOLOR_BUILD_DIR" -ForegroundColor Gray
Write-Host ""

Write-Host "[OK] Odometry-Old" -ForegroundColor Green
Write-Host "  Location: $ODOMETRY_OLD_BUILD_DIR" -ForegroundColor Gray
Write-Host ""

Write-Host "[OK] PGO" -ForegroundColor Green
Write-Host "  Location: $PGO_BUILD_DIR" -ForegroundColor Gray
Write-Host ""

Write-Host "[OK] Python Tools ($success/$($PYTHON_FILES.Count) compiled)" -ForegroundColor Green
Write-Host "  Location: $PYTHON_TOOLS_DIR" -ForegroundColor Gray

if ($success -gt 0) {
    Write-Host ""
    Write-Host "EXE Files:" -ForegroundColor Yellow
    Get-ChildItem -Path $PYTHON_TOOLS_DIR -Filter "*.exe" | ForEach-Object {
        $size = $_.Length / 1MB
        Write-Host "  - $($_.Name) ($([Math]::Round($size, 2)) MB)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All builds completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

exit 0
