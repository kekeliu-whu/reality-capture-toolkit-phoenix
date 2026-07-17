#!/usr/bin/env pwsh
# Comprehensive build script for Colmap, Odometry, and Python Tools
# Builds C++ projects and compiles Python tools to EXE
# Usage: .\build-all.ps1 [-TestOnly]

param(
    [switch]$TestOnly = $false
)

# Project paths
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$VENV_SCRIPTS_DIR = Join-Path $PROJECT_ROOT ".venv\Scripts"
$PYTHON_EXE = Join-Path $VENV_SCRIPTS_DIR "python.exe"
$PYINSTALLER_EXE = Join-Path $VENV_SCRIPTS_DIR "pyinstaller.exe"
$CUDA_ARCHITECTURES = "75-virtual;90-virtual"

# Use project-local Python tools directly instead of relying on shell activation.
Write-Host "Checking Python virtual environment..." -ForegroundColor Yellow
if (!(Test-Path $PYTHON_EXE)) {
    Write-Host "ERROR: Python executable not found at $PYTHON_EXE" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $PYINSTALLER_EXE)) {
    Write-Host "ERROR: PyInstaller executable not found at $PYINSTALLER_EXE" -ForegroundColor Red
    exit 1
}

Write-Host "Python environment ready" -ForegroundColor Green
Write-Host "  Python: $PYTHON_EXE" -ForegroundColor Gray
Write-Host "  PyInstaller: $PYINSTALLER_EXE" -ForegroundColor Gray
Write-Host "  CUDA architectures: $CUDA_ARCHITECTURES" -ForegroundColor Gray
Write-Host ""

# CMake executable path
$CMAKE_EXE = "D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$VCPKG_TOOLCHAIN_FILE = "F:/Library/vcpkg/scripts/buildsystems/vcpkg.cmake"

$BUILD_ALL_DIR = Join-Path $PROJECT_ROOT "build-all"
$COLMAP_SOURCE_DIR = Join-Path $PROJECT_ROOT "colmap"
$COLMAP_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-colmap"
$COLMAP_INSTALL_DIR = Join-Path $BUILD_ALL_DIR "install-colmap"
$COLMAP_CMAKE_DIR = Join-Path $COLMAP_INSTALL_DIR "share\colmap"
$ODOMETRY_SOURCE_DIR = Join-Path $PROJECT_ROOT "odometry"
$ODOMETRY_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-odometry"
$PGO_SOURCE_DIR = Join-Path $PROJECT_ROOT "pgo"
$PGO_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-pgo"
$SCRIPTS_DIR = Join-Path $PROJECT_ROOT "migration\scripts"
$PYTHON_TOOLS_DIR = Join-Path $BUILD_ALL_DIR "build-python-tools"

# ============================================================
# Title and Prerequisites Check
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Complete Build System" -ForegroundColor Cyan
Write-Host "  * Colmap" -ForegroundColor Cyan
Write-Host "  * Odometry" -ForegroundColor Cyan
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
if (!(Test-Path $COLMAP_SOURCE_DIR)) {
    Write-Host "ERROR: Colmap source not found at $COLMAP_SOURCE_DIR" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $ODOMETRY_SOURCE_DIR)) {
    Write-Host "ERROR: Odometry source not found at $ODOMETRY_SOURCE_DIR" -ForegroundColor Red
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
    "insta_compute_pano_poses.py",
    "insta_compute_poses.py",
    "insta_time_sync.py",
    'xsfm_inject_subview_priors.py',
    'xsfm_fix_rig_database.py'
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
Write-Host "Colmap Build: $COLMAP_BUILD_DIR" -ForegroundColor Gray
Write-Host "Odometry Build: $ODOMETRY_BUILD_DIR" -ForegroundColor Gray
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
# Stage 1: Build Colmap
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage 1/3: Building Colmap" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $COLMAP_BUILD_DIR) {
    Write-Host "Removing existing build directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $COLMAP_BUILD_DIR
}

Write-Host "Creating build directory..." -ForegroundColor Gray
New-Item -ItemType Directory -Path $COLMAP_BUILD_DIR -Force | Out-Null

Write-Host "Configuring with CMake..." -ForegroundColor Yellow

& $CMAKE_EXE `
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE `
    "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN_FILE" `
    "-DVCPKG_MANIFEST_MODE=OFF" `
    "-DFETCH_FAISS=OFF" `
    "-DFETCH_ONNX=OFF" `
    "-DONNX_ENABLED=OFF" `
    "-DFETCH_POSELIB=OFF" `
    "-DCUDA_ENABLED=ON" `
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES" `
    --no-warn-unused-cli `
    -S $COLMAP_SOURCE_DIR `
    -B $COLMAP_BUILD_DIR `
    -G "Visual Studio 17 2022" `
    -T "host=x64" `
    -A "x64"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Colmap CMake configuration failed" -ForegroundColor Red
    exit 1
}

Write-Host "Building..." -ForegroundColor Yellow

& $CMAKE_EXE `
    --build $COLMAP_BUILD_DIR `
    --config "Release" `
    --target "ALL_BUILD" `
    -j 24

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Colmap build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Colmap build completed" -ForegroundColor Green
Write-Host ""

Write-Host "Installing Colmap development package..." -ForegroundColor Yellow

& $CMAKE_EXE `
    --install $COLMAP_BUILD_DIR `
    --config "Release" `
    --prefix $COLMAP_INSTALL_DIR

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Colmap install failed" -ForegroundColor Red
    exit 1
}

$COLMAP_CONFIG_FILE = Join-Path $COLMAP_CMAKE_DIR "colmap-config.cmake"
if (!(Test-Path $COLMAP_CONFIG_FILE)) {
    Write-Host "ERROR: Colmap CMake package was not installed at $COLMAP_CONFIG_FILE" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Colmap development package installed" -ForegroundColor Green
Write-Host "  CMake package: $COLMAP_CMAKE_DIR" -ForegroundColor Gray
Write-Host ""

# ============================================================
# Stage 2: Build Odometry
# ============================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stage 2/3: Building Odometry" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $ODOMETRY_BUILD_DIR) {
    Write-Host "Removing existing build directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $ODOMETRY_BUILD_DIR
}

Write-Host "Creating build directory..." -ForegroundColor Gray
New-Item -ItemType Directory -Path $ODOMETRY_BUILD_DIR -Force | Out-Null

Write-Host "Configuring with CMake..." -ForegroundColor Yellow

& $CMAKE_EXE `
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE `
    "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN_FILE" `
    --no-warn-unused-cli `
    -S $ODOMETRY_SOURCE_DIR `
    -B $ODOMETRY_BUILD_DIR `
    -G "Visual Studio 17 2022" `
    -T "host=x64" `
    -A "x64"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Odometry CMake configuration failed" -ForegroundColor Red
    exit 1
}

Write-Host "Building..." -ForegroundColor Yellow

& $CMAKE_EXE `
    --build $ODOMETRY_BUILD_DIR `
    --config "Release" `
    --target "ALL_BUILD" `
    -j 24

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Odometry build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Odometry build completed" -ForegroundColor Green
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
    "-Dcolmap_DIR=$COLMAP_CMAKE_DIR" `
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES" `
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

# Generate a single PyInstaller spec that puts all EXEs into one shared bundle.
# Using exclude_binaries=True + single COLLECT deduplicates _internal across all EXEs.
$specLines = @'
# -*- mode: python ; coding: utf-8 -*-
# Auto-generated by build-all.ps1 — do not edit manually.

_scripts = [
    'insta_data_extraction',
    'insta_compute_pano_poses',
    'insta_compute_poses',
    'insta_time_sync',
    'xsfm_inject_subview_priors',
    'xsfm_fix_rig_database'
]

_analyses = []
for _name in _scripts:
    _a = Analysis(
        [_name + '.py'],
        pathex=['.'],
        binaries=[],
        datas=[],
        hiddenimports=[],
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=[],
        noarchive=False,
    )
    _analyses.append((_name, _a))

_collect_args = []
for _name, _a in _analyses:
    _pyz = PYZ(_a.pure)
    _exe = EXE(
        _pyz,
        _a.scripts,
        [],
        exclude_binaries=True,
        name=_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
    )
    _collect_args.extend([_exe, _a.binaries, _a.datas])

coll = COLLECT(
    *_collect_args,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='python_tools',
)
'@

$specFile = Join-Path $SCRIPTS_DIR "python_tools_shared.spec"
$specLines | Set-Content -Path $specFile -Encoding utf8

Write-Host "Compiling all Python tools into shared bundle..." -ForegroundColor Yellow
Write-Host ""

& $PYINSTALLER_EXE $specFile -y
$exit_code = $LASTEXITCODE

$bundle_dist = Join-Path $SCRIPTS_DIR "dist\python_tools"
$success = 0

if ($exit_code -eq 0 -and (Test-Path $bundle_dist)) {
    if (Test-Path $PYTHON_TOOLS_DIR) { Remove-Item $PYTHON_TOOLS_DIR -Recurse -Force }
    Move-Item $bundle_dist $PYTHON_TOOLS_DIR -Force
    $success = $PYTHON_FILES.Count
    Write-Host "  [OK] Shared bundle created" -ForegroundColor Green
} else {
    Write-Host "  [FAILED] (exit code: $exit_code)" -ForegroundColor Red
}

# Cleanup
Write-Host ""
Write-Host "Cleaning up temporary files..." -ForegroundColor Gray
Remove-Item (Join-Path $SCRIPTS_DIR "build"), (Join-Path $SCRIPTS_DIR "dist"), $specFile -Recurse -Force -ErrorAction SilentlyContinue

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

Write-Host "[OK] Colmap" -ForegroundColor Green
Write-Host "  Location: $COLMAP_BUILD_DIR" -ForegroundColor Gray
Write-Host ""

Write-Host "[OK] Odometry" -ForegroundColor Green
Write-Host "  Location: $ODOMETRY_BUILD_DIR" -ForegroundColor Gray
Write-Host ""

Write-Host "[OK] PGO" -ForegroundColor Green
Write-Host "  Location: $PGO_BUILD_DIR" -ForegroundColor Gray
Write-Host ""

Write-Host "[OK] Python Tools ($success/$($PYTHON_FILES.Count) compiled)" -ForegroundColor Green
Write-Host "  Location: $PYTHON_TOOLS_DIR" -ForegroundColor Gray

if ($success -gt 0) {
    Write-Host ""
    $totalMB = (Get-ChildItem $PYTHON_TOOLS_DIR -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "  Bundle size: $([Math]::Round($totalMB, 2)) MB" -ForegroundColor Gray
    Write-Host "  EXEs:" -ForegroundColor Yellow
    Get-ChildItem -Path $PYTHON_TOOLS_DIR -Filter "*.exe" | ForEach-Object {
        Write-Host "    - $($_.Name)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All builds completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

exit 0
