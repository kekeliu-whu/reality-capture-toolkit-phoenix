# =====================================
# Build Pack Script
# =====================================
# Create a distribution package with all EXE files and dependencies
# Output: build-pack folder containing all binaries and DLLs

param(
    [switch]$TestOnly = $false,
    [switch]$Clean = $false
)

# =====================================
# Configuration
# =====================================
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$PACK_DIR = Join-Path $PROJECT_ROOT "build-pack"
$BUILD_ALL_DIR = Join-Path $PROJECT_ROOT "build-all"

# Build output paths
$XCOLOR_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-xcolor\Release"
$XCOLOR_MIGRATION_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-xcolor\migration_build\Release"
$ODOMETRY_OLD_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-odometry-old\Release"
$PYTHON_TOOLS_DIR = Join-Path $BUILD_ALL_DIR "build-python-tools"

# CUDA paths
$CUDA_BIN_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

# Data files
$PROJ_DB_SOURCE = Join-Path $XCOLOR_BUILD_DIR "proj.db"

# =====================================
# Functions
# =====================================
function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "========================================"
    Write-Host $Text
    Write-Host "========================================"
}

function Write-Status {
    param([string]$Text)
    Write-Host "  [OK] $Text" -ForegroundColor Green
}

function Write-ErrorMsg {
    param([string]$Text)
    Write-Host "  [ERROR] $Text" -ForegroundColor Red
}

function Test-PrerequisitesAndBuild {
    Write-Section "Checking prerequisites and build artifacts"

    $missing = @()

    # Check build-all directory
    if (-not (Test-Path $BUILD_ALL_DIR)) {
        $missing += "build-all directory not found (run build-all.ps1 first)"
    }

    # Verify specific executable files from XColor
    Write-Status "Checking XColor executables..."
    $xcolorExes = @("xcolor_main.exe", "xsfm.exe", "xsfm_pre.exe", "xsfm_process_point_cloud.exe")
    $xcolorExeCount = 0
    foreach ($exe in $xcolorExes) {
        if (Test-Path (Join-Path $XCOLOR_BUILD_DIR $exe)) {
            $xcolorExeCount++
        }
    }
    if ($xcolorExeCount -eq 0) {
        $missing += "No XColor executables found"
    } else {
        Write-Status "XColor executables found: $xcolorExeCount"
    }

    # Verify specific executable from Odometry-Old
    Write-Status "Checking Odometry-Old executables..."
    if (-not (Test-Path (Join-Path $ODOMETRY_OLD_BUILD_DIR "slam_core_main.exe"))) {
        $missing += "slam_core_main.exe not found"
    } else {
        Write-Status "slam_core_main.exe found"
    }

    # Verify Python Tools
    Write-Status "Checking Python Tools executables..."
    $pythonExes = @("insta_compute_poses.exe", "insta_data_extraction.exe", "insta_time_sync.exe")
    $pythonExeCount = 0
    foreach ($exe in $pythonExes) {
        if (Test-Path (Join-Path $PYTHON_TOOLS_DIR $exe)) {
            $pythonExeCount++
        }
    }
    if ($pythonExeCount -eq 0) {
        $missing += "No Python Tools executables found"
    } else {
        Write-Status "Python Tools executables found: $pythonExeCount"
    }

    # Verify convert_s20.exe from migration_build
    Write-Status "Checking XColor migration_build executables..."
    if (-not (Test-Path (Join-Path $XCOLOR_MIGRATION_BUILD_DIR "convert_s20.exe"))) {
        $missing += "convert_s20.exe not found in migration_build"
    } else {
        Write-Status "convert_s20.exe found"
    }

    # Check build-all Release directories for DLL dependencies
    $xcolorDllCount = @(Get-ChildItem $XCOLOR_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    $xcolorMigrationDllCount = @(Get-ChildItem $XCOLOR_MIGRATION_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    $odometryDllCount = @(Get-ChildItem $ODOMETRY_OLD_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    $totalDllCount = $xcolorDllCount + $xcolorMigrationDllCount + $odometryDllCount
    
    if ($totalDllCount -eq 0) {
        $missing += "No DLL files found in Release directories"
    } else {
        Write-Status "Dependency DLL files: $totalDllCount (XColor: $xcolorDllCount, XColor Migration: $xcolorMigrationDllCount, Odometry-Old: $odometryDllCount)"
    }

    # Check CUDA DLL files
    Write-Status "Checking CUDA DLL files..."
    if (-not (Test-Path $CUDA_BIN_DIR)) {
        Write-Host "    WARNING: CUDA bin directory not found at $CUDA_BIN_DIR" -ForegroundColor Yellow
    } else {
        $cudaDllCount = @(Get-ChildItem $CUDA_BIN_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
        if ($cudaDllCount -eq 0) {
            Write-Host "    WARNING: No DLL files found in CUDA bin directory" -ForegroundColor Yellow
        } else {
            Write-Status "CUDA DLL files found: $cudaDllCount"
        }
    }

    # Check proj.db file
    Write-Status "Checking proj.db file..."
    if (-not (Test-Path $PROJ_DB_SOURCE)) {
        Write-Host "    WARNING: proj.db not found at $PROJ_DB_SOURCE" -ForegroundColor Yellow
    } else {
        Write-Status "proj.db file found"
    }

    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-ErrorMsg "Missing components:"
        foreach ($item in $missing) {
            Write-Host "    - $item" -ForegroundColor Red
        }
        return $false
    }

    Write-Status "All prerequisites check passed"
    return $true
}

function Create-PackDirectory {
    Write-Section "Creating pack directory"

    if (Test-Path $PACK_DIR) {
        Write-Status "Removing existing build-pack directory..."
        Remove-Item -Recurse -Force $PACK_DIR
    }

    New-Item -ItemType Directory -Path $PACK_DIR -Force | Out-Null
    Write-Status "build-pack directory created: $PACK_DIR"
}

function Copy-ExecutableFiles {
    Write-Section "Copying executable files"

    # List of executable files to copy (as documented in README)
    $exeFiles = @(
        # XColor Component
        "xcolor_main.exe",
        "xsfm.exe",
        "xsfm_image_sampler.exe",
        "xsfm_pre.exe",
        "xsfm_process_point_cloud.exe",
        "xsfm_reset_cameras.exe",
        "crashpad_handler.exe",
        # XColor Migration Build
        "convert_s20.exe",
        # Odometry-Old Component
        "slam_core_main.exe",
        # Python Tools
        "insta_compute_poses.exe",
        "insta_data_extraction.exe",
        "insta_time_sync.exe"
    )

    Write-Status "Copying documented executable files..."
    $copiedCount = 0
    
    # Search for each executable in all build directories
    $allExeFiles = Get-ChildItem $XCOLOR_BUILD_DIR -Filter "*.exe" -ErrorAction SilentlyContinue
    $allExeFiles += Get-ChildItem $XCOLOR_MIGRATION_BUILD_DIR -Filter "*.exe" -ErrorAction SilentlyContinue
    $allExeFiles += Get-ChildItem $ODOMETRY_OLD_BUILD_DIR -Filter "*.exe" -ErrorAction SilentlyContinue
    $allExeFiles += Get-ChildItem $PYTHON_TOOLS_DIR -Filter "*.exe" -ErrorAction SilentlyContinue

    foreach ($exeName in $exeFiles) {
        $foundFile = $allExeFiles | Where-Object { $_.Name -eq $exeName }
        if ($foundFile) {
            Copy-Item -LiteralPath $foundFile.FullName -Destination $PACK_DIR -Force
            Write-Host "    Copied: $exeName"
            $copiedCount++
        } else {
            Write-Host "    WARNING: Not found: $exeName" -ForegroundColor Yellow
        }
    }
    
    Write-Host "    Total copied: $copiedCount files"
}

function Copy-Dependencies {
    Write-Section "Copying dependencies from Release directories"

    Write-Status "Copying XColor Release DLL files..."
    Get-ChildItem $XCOLOR_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $PACK_DIR -Force
    }
    $count = @(Get-ChildItem $XCOLOR_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    Write-Host "    Copied $count files"

    Write-Status "Copying XColor migration_build Release DLL files..."
    Get-ChildItem $XCOLOR_MIGRATION_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $PACK_DIR -Force
    }
    $count = @(Get-ChildItem $XCOLOR_MIGRATION_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    Write-Host "    Copied $count files"

    Write-Status "Copying Odometry-Old Release DLL files..."
    Get-ChildItem $ODOMETRY_OLD_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $PACK_DIR -Force
    }
    $count = @(Get-ChildItem $ODOMETRY_OLD_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    Write-Host "    Copied $count files"

    Write-Status "Copying CUDA DLL files..."
    if (Test-Path $CUDA_BIN_DIR) {
        Get-ChildItem $CUDA_BIN_DIR -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $PACK_DIR -Force
        }
        $count = @(Get-ChildItem $CUDA_BIN_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
        Write-Host "    Copied $count files"
    } else {
        Write-Host "    WARNING: CUDA directory not found at $CUDA_BIN_DIR" -ForegroundColor Yellow
    }
}

function Copy-DataFiles {
    Write-Section "Copying data files"

    Write-Status "Copying proj.db..."
    if (Test-Path $PROJ_DB_SOURCE) {
        Copy-Item -LiteralPath $PROJ_DB_SOURCE -Destination $PACK_DIR -Force
        Write-Host "    Copied proj.db"
    } else {
        Write-Host "    WARNING: proj.db not found at $PROJ_DB_SOURCE" -ForegroundColor Yellow
    }
}

function Show-PackSummary {
    Write-Section "Package Summary"

    $exeCount = @(Get-ChildItem $PACK_DIR -Filter "*.exe" -ErrorAction SilentlyContinue).Count
    $dllCount = @(Get-ChildItem $PACK_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    $totalFiles = @(Get-ChildItem $PACK_DIR -Recurse -ErrorAction SilentlyContinue).Count
    $totalSize = (Get-ChildItem -Path $PACK_DIR -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $totalSizeMB = if ($totalSize) { [Math]::Round($totalSize / 1MB, 2) } else { 0 }

    Write-Host ""
    Write-Host "  Total files:     $totalFiles"
    Write-Host "  EXE files:       $exeCount"
    Write-Host "  DLL files:       $dllCount"
    Write-Host "  Total size:      $totalSizeMB MB"
    Write-Host ""
    Write-Host "  Location: $PACK_DIR" -ForegroundColor Yellow
    Write-Host ""
}

# =====================================
# Main
# =====================================
Write-Host ""
Write-Host "===================================================="
Write-Host "         Build Pack Script"
Write-Host "  Create distribution package (EXE + DLL + Config)"
Write-Host "===================================================="
Write-Host ""

if ($TestOnly) {
    Write-Host "[TEST MODE]" -ForegroundColor Yellow
    Write-Host ""
}

# Check prerequisites
if (-not (Test-PrerequisitesAndBuild)) {
    Write-Host ""
    Write-ErrorMsg "Prerequisites check failed"
    exit 1
}

if ($TestOnly) {
    Write-Section "Test Result"
    Write-Status "All checks passed - ready to pack"
    Write-Host ""
    exit 0
}

# Create pack directory
Create-PackDirectory

# Copy files
Copy-ExecutableFiles
Copy-Dependencies
Copy-DataFiles

# Show summary
Show-PackSummary

Write-Host "COMPLETED!" -ForegroundColor Green
Write-Host ""
