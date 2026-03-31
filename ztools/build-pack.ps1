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
$PGO_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-pgo\Release"
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

function Fail {
    param([string]$Text)
    Write-ErrorMsg $Text
    exit 1
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
    $xcolorExes = @("xcolor.exe", "xsfm.exe", "xsfm_pre.exe", "xsfm_process_point_cloud.exe")
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
    if (-not (Test-Path (Join-Path $ODOMETRY_OLD_BUILD_DIR "slam.exe"))) {
        $missing += "slam.exe not found"
    } else {
        Write-Status "slam.exe found"
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

    # Verify convert_manifold.exe from migration_build
    Write-Status "Checking XColor migration_build executables..."
    if (-not (Test-Path (Join-Path $XCOLOR_MIGRATION_BUILD_DIR "convert_manifold.exe"))) {
        $missing += "convert_manifold.exe not found in migration_build"
    } else {
        Write-Status "convert_manifold.exe found"
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
        "xcolor.exe",
        "xsfm.exe",
        "xsfm_image_sampler.exe",
        "xsfm_pre.exe",
        "xsfm_process_point_cloud.exe",
        "xsfm_reset_cameras.exe",
        "crashpad_handler.exe",
        # XColor Migration Build
        "convert_manifold.exe",
        # Odometry-Old Component
        "slam.exe",
        # PGO Component
        "slam_post.exe",
        # Python Tools
        "insta_compute_poses.exe",
        "insta_data_extraction.exe",
        "insta_time_sync.exe"
    )

    Write-Status "Copying documented executable files..."
    $copiedCount = 0
    $missing = @()
    
    # Search for each executable in all build directories
    $allExeFiles = Get-ChildItem $XCOLOR_BUILD_DIR -Filter "*.exe" -ErrorAction SilentlyContinue
    $allExeFiles += Get-ChildItem $XCOLOR_MIGRATION_BUILD_DIR -Filter "*.exe" -ErrorAction SilentlyContinue
    $allExeFiles += Get-ChildItem $ODOMETRY_OLD_BUILD_DIR -Filter "*.exe" -ErrorAction SilentlyContinue
    $allExeFiles += Get-ChildItem $PGO_BUILD_DIR -Filter "*.exe" -ErrorAction SilentlyContinue
    $allExeFiles += Get-ChildItem $PYTHON_TOOLS_DIR -Filter "*.exe" -ErrorAction SilentlyContinue

    foreach ($exeName in $exeFiles) {
        $foundFile = $allExeFiles | Where-Object { $_.Name -eq $exeName }
        if ($foundFile) {
            try {
                Copy-Item -LiteralPath $foundFile.FullName -Destination $PACK_DIR -Force -ErrorAction Stop
                Write-Host "    Copied: $exeName"
                $copiedCount++
            } catch {
                $errMsg = $_.Exception.Message
                Fail "Failed to copy executable ${exeName}: ${errMsg}"
            }
        } else {
            $missing += $exeName
        }
    }

    if ($missing.Count -gt 0) {
        Fail "Missing executable files: $($missing -join ', ')"
    }

    Write-Host "    Total copied: $copiedCount files"
}

function Copy-Dependencies {
    Write-Section "Copying dependencies from Release directories"

    Write-Status "Copying XColor Release DLL files..."
    $xcolorDlls = Get-ChildItem $XCOLOR_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($xcolorDlls.Count -eq 0) { Fail "No DLL files found in $XCOLOR_BUILD_DIR" }
    foreach ($dll in $xcolorDlls) {
        try { Copy-Item -LiteralPath $dll.FullName -Destination $PACK_DIR -Force -ErrorAction Stop } catch { Fail "Failed to copy $($dll.Name): $($_.Exception.Message)" }
    }
    Write-Host "    Copied $($xcolorDlls.Count) files"

    Write-Status "Copying XColor migration_build Release DLL files..."
    $migrationDlls = Get-ChildItem $XCOLOR_MIGRATION_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($migrationDlls.Count -eq 0) { Fail "No DLL files found in $XCOLOR_MIGRATION_BUILD_DIR" }
    foreach ($dll in $migrationDlls) {
        try { Copy-Item -LiteralPath $dll.FullName -Destination $PACK_DIR -Force -ErrorAction Stop } catch { Fail "Failed to copy $($dll.Name): $($_.Exception.Message)" }
    }
    Write-Host "    Copied $($migrationDlls.Count) files"

    Write-Status "Copying Odometry-Old Release DLL files..."
    $odometryDlls = Get-ChildItem $ODOMETRY_OLD_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($odometryDlls.Count -eq 0) { Fail "No DLL files found in $ODOMETRY_OLD_BUILD_DIR" }
    foreach ($dll in $odometryDlls) {
        try { Copy-Item -LiteralPath $dll.FullName -Destination $PACK_DIR -Force -ErrorAction Stop } catch { Fail "Failed to copy $($dll.Name): $($_.Exception.Message)" }
    }
    Write-Host "    Copied $($odometryDlls.Count) files"

    Write-Status "Copying PGO Release DLL files..."
    if (-not (Test-Path $PGO_BUILD_DIR)) { Fail "PGO build directory not found: $PGO_BUILD_DIR" }
    $pgoDlls = Get-ChildItem $PGO_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($pgoDlls.Count -eq 0) { Fail "No DLL files found in $PGO_BUILD_DIR" }
    foreach ($dll in $pgoDlls) {
        try { Copy-Item -LiteralPath $dll.FullName -Destination $PACK_DIR -Force -ErrorAction Stop } catch { Fail "Failed to copy $($dll.Name): $($_.Exception.Message)" }
    }
    Write-Host "    Copied $($pgoDlls.Count) files"

    Write-Status "Copying CUDA DLL files..."
    if (-not (Test-Path $CUDA_BIN_DIR)) { Fail "CUDA directory not found: $CUDA_BIN_DIR" }
    $cudaDlls = Get-ChildItem $CUDA_BIN_DIR -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($cudaDlls.Count -eq 0) { Fail "No CUDA DLL files found in $CUDA_BIN_DIR" }
    foreach ($dll in $cudaDlls) {
        try { Copy-Item -LiteralPath $dll.FullName -Destination $PACK_DIR -Force -ErrorAction Stop } catch { Fail "Failed to copy $($dll.Name): $($_.Exception.Message)" }
    }
    Write-Host "    Copied $($cudaDlls.Count) files"
}

function Download-VocabTree {
    Write-Section "Downloading vocab_tree_faiss_flickr100K_words32K.bin"

    $url = "https://kompflight.com/d3captureinstaller/vocab_tree_faiss_flickr100K_words32K.bin"
    $destination = Join-Path $PACK_DIR "vocab_tree_faiss_flickr100K_words32K.bin"

    try {
        if (-not (Test-Path $PACK_DIR)) {
            New-Item -ItemType Directory -Path $PACK_DIR -Force | Out-Null
        }

        $proxy = $env:HTTPS_PROXY
        if (-not $proxy) { $proxy = $env:https_proxy }
        if (-not $proxy) { $proxy = $env:HTTP_PROXY }
        if (-not $proxy) { $proxy = $env:http_proxy }

        $invokeParams = @{ Uri = $url; OutFile = $destination; UseBasicParsing = $true; ErrorAction = 'Stop' }
        if ($proxy) {
            Write-Status "Using proxy: $proxy"
            $invokeParams.Proxy = $proxy
            # 允许基于系统凭据认证代理（可选）
            $invokeParams.ProxyUseDefaultCredentials = $true
        }

        Invoke-WebRequest @invokeParams
        Write-Status "Downloaded vocab tree file to $destination"
    } catch {
        Write-ErrorMsg "Failed to download vocab tree file from $url"
        Write-ErrorMsg $_.Exception.Message
        exit 1
    }
}

function Copy-DataFiles {
    Write-Section "Copying data files"

    Write-Status "Copying proj.db..."
    if (-not (Test-Path $PROJ_DB_SOURCE)) { Fail "proj.db not found at $PROJ_DB_SOURCE" }
    try {
        Copy-Item -LiteralPath $PROJ_DB_SOURCE -Destination $PACK_DIR -Force -ErrorAction Stop
        Write-Host "    Copied proj.db"
    } catch {
        Fail "Failed to copy proj.db: $($_.Exception.Message)"
    }

    $pgoJsonSource = Join-Path $PROJECT_ROOT "migration\config\pgo\pgo.json"
    Write-Status "Copying pgo.json..."
    if (-not (Test-Path $pgoJsonSource)) { Fail "pgo.json not found at $pgoJsonSource" }
    try {
        Copy-Item -LiteralPath $pgoJsonSource -Destination $PACK_DIR -Force -ErrorAction Stop
        Write-Host "    Copied pgo.json"
    } catch {
        Fail "Failed to copy pgo.json: $($_.Exception.Message)"
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

# Download vocab tree file first (if missing, this exits)
Download-VocabTree

# Copy files
Copy-ExecutableFiles
Copy-Dependencies
Copy-DataFiles

# Show summary
Show-PackSummary

Write-Host "COMPLETED!" -ForegroundColor Green
Write-Host ""
