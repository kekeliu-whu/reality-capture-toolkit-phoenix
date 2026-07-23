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
$COLMAP_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-colmap"
$COLMAP_EXE_SOURCE_DIR = Join-Path $COLMAP_BUILD_DIR "src\colmap\exe\Release"
$ODOMETRY_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-odometry"
$PGO_BUILD_DIR = Join-Path $BUILD_ALL_DIR "build-pgo"
$PYTHON_TOOLS_DIR = Join-Path $BUILD_ALL_DIR "build-python-tools"

# CUDA paths
$CUDA_BIN_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

# Data files
$PROJ_DB_CANDIDATES = @(
    (Join-Path $PROJECT_ROOT "build-all\build-pgo\Release\proj.db"),
    (Join-Path $PROJECT_ROOT "build\Release\proj.db"),
    (Join-Path $PROJECT_ROOT "build\RelWithDebInfo\proj.db"),
    (Join-Path $PROJECT_ROOT "build\Debug\proj.db"),
    (Join-Path $PROJECT_ROOT "build\proj.db"),
    (Join-Path $PROJECT_ROOT "build-pack\proj.db")
)

# Build directory discovery
$XCOLOR_BUILD_DIRS = @(
    (Join-Path $PGO_BUILD_DIR "Release"),
    (Join-Path $COLMAP_BUILD_DIR "src\colmap\exe\Release"),
    (Join-Path $COLMAP_BUILD_DIR "Release")
)
$XCOLOR_MIGRATION_BUILD_DIRS = @(
    (Join-Path $ODOMETRY_BUILD_DIR "migration_build\Release"),
    (Join-Path $PGO_BUILD_DIR "migration_build\Release")
)
$ODOMETRY_BUILD_DIRS = @(
    (Join-Path $ODOMETRY_BUILD_DIR "Release")
)
$PGO_BUILD_DIRS = @(
    (Join-Path $PGO_BUILD_DIR "Release")
)
$REALTIME_MAPPING_BUILD_DIRS = @(
    (Join-Path $BUILD_ALL_DIR "build-realtime-mapping\Release"),
    (Join-Path $PROJECT_ROOT "build-realtime-mapping-w5\Release"),
    (Join-Path $PROJECT_ROOT "build-realtime-mapping\Release"),
    (Join-Path $PROJECT_ROOT "build-realtime-mapping-asan\Release")
)

# Executable requirements (required first, then optional)
$REQUIRED_EXECUTABLES = @(
    "xcolor.exe",
    "xsfm_process_point_cloud.exe",
    "xsfm_pre.exe",
    "slam.exe",
    "slam_post.exe",
    "REALTIME_MAPPING.exe",
    "insta_compute_pano_poses.exe",
    "insta_compute_poses.exe",
    "insta_data_extraction.exe",
    "insta_time_sync.exe",
    "xsfm_fix_rig_database.exe",
    "xsfm_inject_subview_priors.exe",
    "xsfm_create_initial_model.exe",
    "convert_manifold.exe",
    "convert_hs.exe",
    "crashpad_handler.exe",
    "xsfm_post.exe"
)
$OPTIONAL_EXECUTABLES = @()
$EXECUTABLE_SOURCE_DIR_OVERRIDES = @{
    "xsfm_post.exe" = $PGO_BUILD_DIRS
}
$PYTHON_TOOL_EXECUTABLES = @(
    "insta_data_extraction.exe",
    "insta_compute_pano_poses.exe",
    "insta_compute_poses.exe",
    "insta_time_sync.exe",
    "xsfm_inject_subview_priors.exe",
    "xsfm_fix_rig_database.exe",
    "xsfm_create_initial_model.exe"
)
$PYTHON_INTERNAL_REQUIRED_DIRS = @(
    "contourpy",
    "cryptography",
    "cryptography-*.dist-info",
    "dateutil",
    "google",
    "kiwisolver",
    "matplotlib",
    "numpy",
    "numpy-*.dist-info",
    "numpy.libs",
    "PIL",
    "scipy",
    "scipy.libs",
    "telemetry_parser"
)
$script:SELECTED_MIGRATION_BUILD_DIR = $null
$script:SELECTED_REALTIME_MAPPING_BUILD_DIR = $null
$script:SELECTED_PROJ_DB = $null
$script:DUMPBIN_EXE = $null

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

function Get-ExistingPaths {
    param([string[]]$CandidatePaths)
    $paths = @()
    foreach ($path in $CandidatePaths) {
        if ($path -and (Test-Path $path)) {
            $paths += (Resolve-Path $path).Path
        }
    }
    return $paths
}

function Find-Artifact {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$SearchDirs
    )
    foreach ($dir in $SearchDirs) {
        if (-not (Test-Path $dir)) {
            continue
        }
        $candidate = Join-Path $dir $Name
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

function Find-ExecutableArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$SearchDirs
    )

    if ($EXECUTABLE_SOURCE_DIR_OVERRIDES.ContainsKey($Name)) {
        $overrideDirs = Get-ExistingPaths $EXECUTABLE_SOURCE_DIR_OVERRIDES[$Name]
        return Find-Artifact -Name $Name -SearchDirs $overrideDirs
    }

    return Find-Artifact -Name $Name -SearchDirs $SearchDirs
}

function Resolve-CandidatePath {
    param([string[]]$Candidates)
    foreach ($path in $Candidates) {
        if ($path -and (Test-Path $path)) {
            return (Resolve-Path $path).Path
        }
    }
    return $null
}

function Resolve-Dumpbin {
    $command = Get-Command "dumpbin.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $searchRoots = @()
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPaths = @(
            & $vswhere -latest -products * `
                -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                -property installationPath
        )
        foreach ($installPath in $installPaths) {
            if ($installPath) {
                $searchRoots += (Join-Path $installPath "VC\Tools\MSVC")
            }
        }
    }
    $searchRoots += "D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC"

    foreach ($root in $searchRoots | Select-Object -Unique) {
        if (-not (Test-Path $root)) {
            continue
        }
        $candidate = Get-ChildItem $root -Recurse -Filter "dumpbin.exe" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*\bin\Hostx64\x64\dumpbin.exe" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }
    return $null
}

function Get-PeDllDependencies {
    param([Parameter(Mandatory = $true)][string]$Path)

    $output = @(& $script:DUMPBIN_EXE /nologo /dependents $Path 2>&1)
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to inspect PE dependencies for ${Path}: $($output -join ' ')"
    }

    return @(
        $output | ForEach-Object {
            if ($_ -match '^\s+([^\s]+\.dll)\s*$') {
                $matches[1].ToLowerInvariant()
            }
        } | Sort-Object -Unique
    )
}

function Remove-UnusedCudaDlls {
    Write-Status "Detecting required CUDA DLL dependency closure..."

    if (-not $script:DUMPBIN_EXE) {
        Fail "dumpbin.exe was not found; refusing to prune CUDA DLLs without dependency analysis"
    }

    $cudaSourceDlls = @(Get-ChildItem $CUDA_BIN_DIR -Filter "*.dll" -File -ErrorAction Stop)
    $cudaByName = @{}
    foreach ($dll in $cudaSourceDlls) {
        $cudaByName[$dll.Name.ToLowerInvariant()] = $dll.FullName
    }

    $peRoots = @(
        Get-ChildItem $PACK_DIR -Recurse -File -ErrorAction Stop |
            Where-Object {
                $_.Extension.ToLowerInvariant() -in @(".exe", ".dll", ".pyd") -and
                -not $cudaByName.ContainsKey($_.Name.ToLowerInvariant())
            }
    )
    Write-Host "    Inspecting $($peRoots.Count) packaged PE files"

    $requiredCuda = @{}
    $queue = [System.Collections.Generic.Queue[string]]::new()
    foreach ($root in $peRoots) {
        foreach ($dependency in Get-PeDllDependencies -Path $root.FullName) {
            if ($cudaByName.ContainsKey($dependency) -and -not $requiredCuda.ContainsKey($dependency)) {
                $requiredCuda[$dependency] = $true
                $queue.Enqueue($dependency)
            }
        }
    }

    while ($queue.Count -gt 0) {
        $cudaName = $queue.Dequeue()
        foreach ($dependency in Get-PeDllDependencies -Path $cudaByName[$cudaName]) {
            if ($cudaByName.ContainsKey($dependency) -and -not $requiredCuda.ContainsKey($dependency)) {
                $requiredCuda[$dependency] = $true
                $queue.Enqueue($dependency)
            }
        }
    }

    $removedCount = 0
    $removedBytes = 0
    foreach ($dll in $cudaSourceDlls) {
        $name = $dll.Name.ToLowerInvariant()
        $packPath = Join-Path $PACK_DIR $dll.Name
        if (-not $requiredCuda.ContainsKey($name) -and (Test-Path $packPath)) {
            $removedBytes += (Get-Item $packPath).Length
            Remove-Item -LiteralPath $packPath -Force -ErrorAction Stop
            $removedCount++
        }
    }

    $keptNames = @($requiredCuda.Keys | Sort-Object)
    foreach ($name in $keptNames) {
        if (-not (Test-Path (Join-Path $PACK_DIR $name))) {
            Fail "Required CUDA DLL disappeared during pruning: $name"
        }
    }

    Write-Host "    Kept $($keptNames.Count) CUDA DLLs: $($keptNames -join ', ')"
    Write-Host "    Removed $removedCount unused CUDA DLLs ($([Math]::Round($removedBytes / 1MB, 2)) MB)"
}

function Copy-DllSet {
    param(
        [Parameter(Mandatory = $true)][string[]]$SearchDirs,
        [Parameter(Mandatory = $true)][string]$Label,
        [switch]$Required
    )

    $dllCount = 0
    foreach ($dir in $SearchDirs) {
        if (-not (Test-Path $dir)) {
            continue
        }
        $dlls = Get-ChildItem $dir -Filter "*.dll" -ErrorAction SilentlyContinue
        foreach ($dll in $dlls) {
            try {
                Copy-Item -LiteralPath $dll.FullName -Destination $PACK_DIR -Force -ErrorAction Stop
                $dllCount++
            } catch {
                Fail "Failed to copy $($dll.Name): $($_.Exception.Message)"
            }
        }
    }

    if ($Required -and $dllCount -eq 0) {
        Fail "No DLL files found in $Label"
    }

    Write-Host "    Copied $dllCount files"
    return $dllCount
}

function Copy-PythonInternalItem {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$InternalRoot,
        [Parameter(Mandatory = $true)][string]$DestInternalRoot
    )

    $relativePath = [System.IO.Path]::GetRelativePath($InternalRoot, $Source)
    $targetPath = Join-Path $DestInternalRoot $relativePath
    $targetParent = Split-Path -Parent $targetPath
    if ($targetParent -and -not (Test-Path $targetParent)) {
        New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $targetPath -Recurse -Force -ErrorAction Stop
}

function Copy-PythonToolDependencies {
    $pythonInternalDir = Join-Path $PYTHON_TOOLS_DIR "_internal"
    if (-not (Test-Path $pythonInternalDir)) {
        Write-Host "    WARNING: Python tools _internal folder not found at $pythonInternalDir" -ForegroundColor Yellow
        return
    }

    Write-Status "Copying required Python tool dependencies..."
    $destInternalDir = Join-Path $PACK_DIR "_internal"
    if (Test-Path $destInternalDir) { Remove-Item $destInternalDir -Recurse -Force }
    New-Item -ItemType Directory -Path $destInternalDir -Force | Out-Null

    try {
        foreach ($file in Get-ChildItem $pythonInternalDir -File -ErrorAction SilentlyContinue) {
            Copy-PythonInternalItem -Source $file.FullName -InternalRoot $pythonInternalDir -DestInternalRoot $destInternalDir
        }

        foreach ($pattern in $PYTHON_INTERNAL_REQUIRED_DIRS) {
            $items = Get-ChildItem $pythonInternalDir -Directory -Filter $pattern -ErrorAction SilentlyContinue
            foreach ($item in $items) {
                Copy-PythonInternalItem -Source $item.FullName -InternalRoot $pythonInternalDir -DestInternalRoot $destInternalDir
            }
        }

        $internalFileCount = @(Get-ChildItem $destInternalDir -Recurse -File -ErrorAction SilentlyContinue).Count
        $internalSize = (Get-ChildItem $destInternalDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $internalSizeMB = if ($internalSize) { [Math]::Round($internalSize / 1MB, 2) } else { 0 }
        Write-Host "    Copied required _internal dependencies ($internalFileCount files, $internalSizeMB MB)"
    } catch {
        Fail "Failed to copy Python tool dependencies: $($_.Exception.Message)"
    }
}

function Test-PrerequisitesAndBuild {
    Write-Section "Checking prerequisites and build artifacts"

    $missing = @()

    # Check build-all directory
    if (-not (Test-Path $BUILD_ALL_DIR)) {
        $missing += "build-all directory not found (run build-all.ps1 first)"
    }

    $xcolorDirs = Get-ExistingPaths $XCOLOR_BUILD_DIRS
    $odometryDirs = Get-ExistingPaths $ODOMETRY_BUILD_DIRS
    $pgoDirs = Get-ExistingPaths $PGO_BUILD_DIRS
    $realtimeMappingDirs = Get-ExistingPaths $REALTIME_MAPPING_BUILD_DIRS
    $allExeDirs = $xcolorDirs + $odometryDirs + $pgoDirs + $realtimeMappingDirs
    if (Test-Path $PYTHON_TOOLS_DIR) {
        $allExeDirs += (Resolve-Path $PYTHON_TOOLS_DIR).Path
    }
    $migrationDirs = Get-ExistingPaths $XCOLOR_MIGRATION_BUILD_DIRS
    $convertManifold = Find-Artifact -Name "convert_manifold.exe" -SearchDirs $migrationDirs
    if ($convertManifold) {
        $script:SELECTED_MIGRATION_BUILD_DIR = Split-Path -Path $convertManifold -Parent
        $allExeDirs += $script:SELECTED_MIGRATION_BUILD_DIR
        Write-Status "convert_manifold.exe found: $($script:SELECTED_MIGRATION_BUILD_DIR)"
    } else {
        Write-Host "    [WARN] convert_manifold.exe not found in migration_build directories" -ForegroundColor Yellow
    }
    $realtimeMappingExe = Find-Artifact -Name "REALTIME_MAPPING.exe" -SearchDirs $realtimeMappingDirs
    if ($realtimeMappingExe) {
        $script:SELECTED_REALTIME_MAPPING_BUILD_DIR = Split-Path -Path $realtimeMappingExe -Parent
        Write-Status "REALTIME_MAPPING.exe found: $($script:SELECTED_REALTIME_MAPPING_BUILD_DIR)"
    } else {
        Write-Host "    [WARN] REALTIME_MAPPING.exe not found in realtime_mapping build directories" -ForegroundColor Yellow
    }

    Write-Status "Checking Colmap runtime resource files..."
    $colmapRuntimeItems = @(
        (Join-Path $COLMAP_EXE_SOURCE_DIR "colmap.exe"),
        (Join-Path $COLMAP_EXE_SOURCE_DIR "qt.conf"),
        (Join-Path $COLMAP_EXE_SOURCE_DIR "plugins")
    )
    $missingColmapRuntimeItems = @()
    foreach ($path in $colmapRuntimeItems) {
        if (-not (Test-Path $path)) {
            $missingColmapRuntimeItems += $path
        }
    }
    if ($missingColmapRuntimeItems.Count -gt 0) {
        $missing += "Missing required Colmap runtime resource files: $($missingColmapRuntimeItems -join ', ')"
    } else {
        Write-Status "Colmap runtime resources found at: $COLMAP_EXE_SOURCE_DIR"
    }

    Write-Status "Checking required core executables..."
    $missingRequired = @()
    foreach ($exe in $REQUIRED_EXECUTABLES) {
        $foundExe = Find-ExecutableArtifact -Name $exe -SearchDirs $allExeDirs
        if (-not $foundExe) {
            if ($exe -eq "xsfm_pre.exe" -and (Test-Path (Join-Path $COLMAP_EXE_SOURCE_DIR "colmap.exe"))) {
                continue
            }
            $missingRequired += $exe
        }
    }
    if ($missingRequired.Count -gt 0) {
        $missing += "Required executables not found: $($missingRequired -join ', ')"
    } else {
        Write-Status "Required executables found: $($REQUIRED_EXECUTABLES.Count)"
    }

    Write-Status "Checking optional executables..."
    $missingOptional = @()
    $optionalSearchDirs = $allExeDirs
    if ($script:SELECTED_MIGRATION_BUILD_DIR) {
        $optionalSearchDirs += $script:SELECTED_MIGRATION_BUILD_DIR
    }
    foreach ($exe in $OPTIONAL_EXECUTABLES) {
        if (-not (Find-ExecutableArtifact -Name $exe -SearchDirs $optionalSearchDirs)) {
            $missingOptional += $exe
        }
    }
    if ($missingOptional.Count -gt 0) {
        Write-Host "    [WARN] Optional executables not found: $($missingOptional -join ', ')" -ForegroundColor Yellow
    }

    # Check build-all Release directories for DLL dependencies
    Write-Status "Checking DLL dependencies..."
    $xcolorDllCount = 0
    foreach ($dir in $xcolorDirs) {
        $xcolorDllCount += @(Get-ChildItem $dir -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    }
    $odometryDllCount = 0
    foreach ($dir in $odometryDirs) {
        $odometryDllCount += @(Get-ChildItem $dir -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    }
    $pgoDllCount = 0
    foreach ($dir in $pgoDirs) {
        $pgoDllCount += @(Get-ChildItem $dir -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    }
    $realtimeMappingDllCount = 0
    foreach ($dir in $realtimeMappingDirs) {
        $realtimeMappingDllCount += @(Get-ChildItem $dir -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    }
    $migrationDllCount = 0
    if ($script:SELECTED_MIGRATION_BUILD_DIR -and (Test-Path $script:SELECTED_MIGRATION_BUILD_DIR)) {
        $migrationDllCount = @(Get-ChildItem $script:SELECTED_MIGRATION_BUILD_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    }
    $totalDllCount = $xcolorDllCount + $odometryDllCount + $pgoDllCount + $realtimeMappingDllCount
    if ($totalDllCount -eq 0) {
        $missing += "No DLL files found in XColor/Odometry/PGO/Realtime Mapping Release directories"
    } else {
        Write-Status "Dependency DLL files: $totalDllCount (XColor: $xcolorDllCount, Odometry: $odometryDllCount, PGO: $pgoDllCount, Realtime Mapping: $realtimeMappingDllCount)"
    }
    if ($migrationDllCount -gt 0) {
        Write-Status "Migration DLL files found: $migrationDllCount"
    } else {
        Write-Host "    [WARN] No migration DLL files found" -ForegroundColor Yellow
    }
    if ($realtimeMappingDllCount -gt 0) {
        Write-Status "Realtime Mapping DLL files found: $realtimeMappingDllCount"
    } else {
        Write-Host "    [WARN] No Realtime Mapping DLL files found" -ForegroundColor Yellow
    }

    # CUDA DLLs are copied first, then pruned using the packaged PE dependency graph.
    Write-Status "Checking CUDA DLL files..."
    if (-not (Test-Path $CUDA_BIN_DIR)) {
        $missing += "CUDA bin directory not found at $CUDA_BIN_DIR"
    } else {
        $cudaDllCount = @(Get-ChildItem $CUDA_BIN_DIR -Filter "*.dll" -File -ErrorAction SilentlyContinue).Count
        if ($cudaDllCount -eq 0) {
            $missing += "No CUDA DLL files found in $CUDA_BIN_DIR"
        } else {
            Write-Status "CUDA DLL files found: $cudaDllCount"
        }
    }

    Write-Status "Checking PE dependency analyzer..."
    $script:DUMPBIN_EXE = Resolve-Dumpbin
    if (-not $script:DUMPBIN_EXE) {
        $missing += "dumpbin.exe not found; Visual Studio C++ tools are required for CUDA DLL pruning"
    } else {
        Write-Status "PE dependency analyzer found: $script:DUMPBIN_EXE"
    }

    # Check proj.db file
    Write-Status "Checking proj.db file..."
    $script:SELECTED_PROJ_DB = Resolve-CandidatePath -Candidates $PROJ_DB_CANDIDATES
    if (-not $script:SELECTED_PROJ_DB) {
        Write-Host "    [WARN] proj.db not found in expected paths" -ForegroundColor Yellow
    } else {
        Write-Status "proj.db file found: $script:SELECTED_PROJ_DB"
    }

    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-ErrorMsg "Missing mandatory components:"
        foreach ($item in $missing) {
            Write-Host "    - $item" -ForegroundColor Red
        }
        return $false
    }

    Write-Status "All mandatory prerequisites check passed"
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

    Write-Status "Copying executable files..."

    $xcolorDirs = Get-ExistingPaths $XCOLOR_BUILD_DIRS
    $odometryDirs = Get-ExistingPaths $ODOMETRY_BUILD_DIRS
    $pgoDirs = Get-ExistingPaths $PGO_BUILD_DIRS
    $realtimeMappingDirs = Get-ExistingPaths $REALTIME_MAPPING_BUILD_DIRS
    $allExeSearchDirs = $xcolorDirs + $odometryDirs + $pgoDirs + $realtimeMappingDirs
    if (Test-Path $PYTHON_TOOLS_DIR) {
        $allExeSearchDirs += (Resolve-Path $PYTHON_TOOLS_DIR).Path
    }
    if ($script:SELECTED_MIGRATION_BUILD_DIR) {
        $allExeSearchDirs += $script:SELECTED_MIGRATION_BUILD_DIR
    }

    $copiedCount = 0
    $missingRequired = @()
    $missingOptional = @()

    foreach ($exeName in $REQUIRED_EXECUTABLES) {
        $foundFile = Find-ExecutableArtifact -Name $exeName -SearchDirs $allExeSearchDirs
        $usesColmapFallback = $false
        if (-not $foundFile -and $exeName -eq "xsfm_pre.exe") {
            $foundFile = Join-Path $COLMAP_EXE_SOURCE_DIR "colmap.exe"
            $usesColmapFallback = $true
        }
        if ($foundFile) {
            try {
                if ($usesColmapFallback) {
                    $colmapTempTarget = Join-Path $PACK_DIR "colmap.exe"
                    $xsfmPreTarget = Join-Path $PACK_DIR "xsfm_pre.exe"
                    Copy-Item -LiteralPath $foundFile -Destination $colmapTempTarget -Force -ErrorAction Stop
                    Rename-Item -LiteralPath $colmapTempTarget -NewName "xsfm_pre.exe" -Force -ErrorAction Stop
                    Write-Host "    Renamed: colmap.exe -> xsfm_pre.exe"
                } else {
                    Copy-Item -LiteralPath $foundFile -Destination $PACK_DIR -Force -ErrorAction Stop
                    Write-Host "    Copied: $exeName"
                }
                $copiedCount++
            } catch {
                $errMsg = $_.Exception.Message
                Fail "Failed to copy executable ${exeName}: ${errMsg}"
            }
        } else {
            $missingRequired += $exeName
        }
    }

    foreach ($exeName in $OPTIONAL_EXECUTABLES) {
        $foundFile = Find-ExecutableArtifact -Name $exeName -SearchDirs $allExeSearchDirs
        if ($foundFile) {
            try {
                Copy-Item -LiteralPath $foundFile -Destination $PACK_DIR -Force -ErrorAction Stop
                Write-Host "    Copied: $exeName"
                $copiedCount++
            } catch {
                $errMsg = $_.Exception.Message
                Fail "Failed to copy executable ${exeName}: ${errMsg}"
            }
        } else {
            $missingOptional += $exeName
        }
    }

    if ($missingRequired.Count -gt 0) {
        Fail "Missing required executable files: $($missingRequired -join ', ')"
    }
    if ($missingOptional.Count -gt 0) {
        Write-Host "    [WARN] Optional executable files not found: $($missingOptional -join ', ')" -ForegroundColor Yellow
    }

    Write-Host "    Total copied: $copiedCount files"

    Write-Status "Copying Colmap runtime resources from $COLMAP_EXE_SOURCE_DIR ..."
    $colmapQtConfSource = Join-Path $COLMAP_EXE_SOURCE_DIR "qt.conf"
    $colmapQtConfTarget = Join-Path $PACK_DIR "qt.conf"
    $colmapPluginsSource = Join-Path $COLMAP_EXE_SOURCE_DIR "plugins"
    $colmapPluginsTarget = Join-Path $PACK_DIR "plugins"
    $colmapExeSource = Join-Path $COLMAP_EXE_SOURCE_DIR "colmap.exe"
    $colmapTempTarget = Join-Path $PACK_DIR "colmap.exe"
    $xsfmPreTarget = Join-Path $PACK_DIR "xsfm_pre.exe"

    try {
        Copy-Item -LiteralPath $colmapQtConfSource -Destination $colmapQtConfTarget -Force -ErrorAction Stop
        Write-Host "    Copied: qt.conf"
    } catch {
        Fail "Failed to copy qt.conf: $($_.Exception.Message)"
    }

    if (Test-Path $colmapPluginsTarget) {
        Remove-Item -Recurse -Force $colmapPluginsTarget
    }
    try {
        Copy-Item -LiteralPath $colmapPluginsSource -Destination $colmapPluginsTarget -Recurse -Force -ErrorAction Stop
        Write-Host "    Copied: plugins/"
    } catch {
        Fail "Failed to copy plugins directory: $($_.Exception.Message)"
    }

    try {
        Copy-Item -LiteralPath $colmapExeSource -Destination $colmapTempTarget -Force -ErrorAction Stop
        if (Test-Path $xsfmPreTarget) {
            Remove-Item -LiteralPath $xsfmPreTarget -Force -ErrorAction Stop
        }
        Rename-Item -LiteralPath $colmapTempTarget -NewName "xsfm_pre.exe" -Force -ErrorAction Stop
        Write-Host "    Renamed: colmap.exe -> xsfm_pre.exe"
    } catch {
        Fail "Failed to rename colmap.exe as xsfm_pre.exe: $($_.Exception.Message)"
    }

    # Copy ffmpeg.exe alongside the Python tools (used by insta_data_extraction)
    Write-Status "Copying ffmpeg.exe..."
    $ffmpeg = Get-Command "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if (-not $ffmpeg) {
        $commonPaths = @(
            "C:\ffmpeg\bin\ffmpeg.exe",
            "C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg.exe"
        )
        $ffmpeg = $commonPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    }
    if ($ffmpeg) {
        Copy-Item -LiteralPath $ffmpeg -Destination (Join-Path $PACK_DIR "ffmpeg.exe") -Force -ErrorAction Stop
        Write-Host "    Copied: ffmpeg.exe ($ffmpeg)"
    } else {
        Write-Host "    WARNING: ffmpeg.exe not found, please install it and add to PATH" -ForegroundColor Yellow
    }
}

function Copy-Dependencies {
    Write-Section "Copying dependencies from Release directories"

    $xcolorDirs = Get-ExistingPaths $XCOLOR_BUILD_DIRS
    $odometryDirs = Get-ExistingPaths $ODOMETRY_BUILD_DIRS
    $pgoDirs = Get-ExistingPaths $PGO_BUILD_DIRS
    $realtimeMappingDirs = Get-ExistingPaths $REALTIME_MAPPING_BUILD_DIRS

    Write-Status "Copying XColor Release DLL files..."
    $xcolorDllCount = Copy-DllSet -SearchDirs $xcolorDirs -Label "XColor Release" -Required

    if ($script:SELECTED_MIGRATION_BUILD_DIR) {
        Write-Status "Copying XColor migration_build Release DLL files..."
        $migrationDllCount = Copy-DllSet -SearchDirs @($script:SELECTED_MIGRATION_BUILD_DIR) -Label "XColor migration_build" -Required:$false
        if ($migrationDllCount -eq 0) {
            Write-Host "    [WARN] No DLLs found in migration_build release directory" -ForegroundColor Yellow
        }
    } else {
        Write-Host "    [WARN] Migration build directory not selected, skipping migration DLL copy" -ForegroundColor Yellow
    }

    Write-Status "Copying Odometry Release DLL files..."
    $odometryDllCount = Copy-DllSet -SearchDirs $odometryDirs -Label "Odometry Release" -Required

    Write-Status "Copying PGO Release DLL files..."
    $pgoDllCount = Copy-DllSet -SearchDirs $pgoDirs -Label "PGO Release" -Required

    Write-Status "Copying Realtime Mapping Release DLL files..."
    $realtimeMappingDllCount = Copy-DllSet -SearchDirs $realtimeMappingDirs -Label "Realtime Mapping Release" -Required

    Write-Status "Copying CUDA DLL files..."
    if (-not (Test-Path $CUDA_BIN_DIR)) { Fail "CUDA directory not found: $CUDA_BIN_DIR" }
    $cudaDlls = @(Get-ChildItem $CUDA_BIN_DIR -Filter "*.dll" -File -ErrorAction Stop)
    if ($cudaDlls.Count -eq 0) { Fail "No CUDA DLL files found in $CUDA_BIN_DIR" }
    foreach ($dll in $cudaDlls) {
        try { Copy-Item -LiteralPath $dll.FullName -Destination $PACK_DIR -Force -ErrorAction Stop } catch { Fail "Failed to copy $($dll.Name): $($_.Exception.Message)" }
    }
    Write-Host "    Copied $($cudaDlls.Count) files"

    if ($xcolorDllCount -eq 0 -or $odometryDllCount -eq 0 -or $pgoDllCount -eq 0 -or $realtimeMappingDllCount -eq 0) {
        Fail "Required DLL copy failed for one or more build components"
    }

    Copy-PythonToolDependencies
    Remove-UnusedCudaDlls
}

function Download-VocabTree {
    Write-Section "Downloading vocab_tree_faiss_flickr100K_words32K.bin"

    $url = "https://kompflight.com/d3captureinstaller/vocab_tree_faiss_flickr100K_words32K.bin"
    $destination = Join-Path $PACK_DIR "vocab_tree_faiss_flickr100K_words32K.bin"

    try {
        if (-not (Test-Path $PACK_DIR)) {
            New-Item -ItemType Directory -Path $PACK_DIR -Force | Out-Null
        }

        $invokeParams = @{
            Uri = $url
            OutFile = $destination
            UseBasicParsing = $true
            NoProxy = $true
            ErrorAction = 'Stop'
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
    if (-not $script:SELECTED_PROJ_DB) {
        $script:SELECTED_PROJ_DB = Resolve-CandidatePath -Candidates $PROJ_DB_CANDIDATES
    }

    if (-not $script:SELECTED_PROJ_DB) {
        Write-Host "    [WARN] proj.db not found, continuing without it" -ForegroundColor Yellow
    } else {
        try {
            Copy-Item -LiteralPath $script:SELECTED_PROJ_DB -Destination $PACK_DIR -Force -ErrorAction Stop
            Write-Host "    Copied proj.db"
        } catch {
            Fail "Failed to copy proj.db: $($_.Exception.Message)"
        }
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

    $l2ProYamlSource = $null
    if ($script:SELECTED_REALTIME_MAPPING_BUILD_DIR) {
        $candidate = Join-Path $script:SELECTED_REALTIME_MAPPING_BUILD_DIR "L2PRO.yaml"
        if (Test-Path $candidate) {
            $l2ProYamlSource = $candidate
        }
    }
    if (-not $l2ProYamlSource) {
        $l2ProYamlSource = Resolve-CandidatePath -Candidates @(
            (Get-ExistingPaths $REALTIME_MAPPING_BUILD_DIRS | ForEach-Object { Join-Path $_ "L2PRO.yaml" }),
            (Join-Path $PROJECT_ROOT "realtime_mapping\lio_core_ros\config\L2PRO.yaml")
        )
    }

    Write-Status "Copying L2PRO.yaml..."
    if (-not $l2ProYamlSource) { Fail "L2PRO.yaml not found in realtime_mapping config directories" }
    try {
        Copy-Item -LiteralPath $l2ProYamlSource -Destination $PACK_DIR -Force -ErrorAction Stop
        Write-Host "    Copied L2PRO.yaml"
    } catch {
        Fail "Failed to copy L2PRO.yaml: $($_.Exception.Message)"
    }
}

function Show-PackSummary {
    Write-Section "Package Summary"

    $exeCount = @(Get-ChildItem $PACK_DIR -Filter "*.exe" -ErrorAction SilentlyContinue).Count
    $dllCount = @(Get-ChildItem $PACK_DIR -Filter "*.dll" -ErrorAction SilentlyContinue).Count
    $totalFiles = @(Get-ChildItem $PACK_DIR -Recurse -ErrorAction SilentlyContinue).Count
    $totalSize = (Get-ChildItem -Path $PACK_DIR -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $totalSizeMB = if ($totalSize) { [Math]::Round($totalSize / 1MB, 2) } else { 0 }

    # Check for Python tools shared bundle
    $pythonInternalDir = Join-Path $PACK_DIR "_internal"
    $hasPythonBundle = Test-Path $pythonInternalDir
    if ($hasPythonBundle) {
        $internalFileCount = @(Get-ChildItem $pythonInternalDir -Recurse -File -ErrorAction SilentlyContinue).Count
        $internalSize = (Get-ChildItem $pythonInternalDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $internalSizeMB = if ($internalSize) { [Math]::Round($internalSize / 1MB, 2) } else { 0 }
    }

    Write-Host ""
    Write-Host "  Total files:     $totalFiles"
    Write-Host "  EXE files:       $exeCount"
    Write-Host "  DLL files:       $dllCount"
    $hasFfmpeg = Test-Path (Join-Path $PACK_DIR "ffmpeg.exe")
    if ($hasFfmpeg) {
        Write-Host "  ffmpeg.exe:      bundled" -ForegroundColor Green
    }
    if ($hasPythonBundle) {
        Write-Host "  Python bundle:   _internal/ ($internalFileCount files, $internalSizeMB MB)"
    }
    Write-Host "  Total size:      $totalSizeMB MB"
    Write-Host ""
    Write-Host "  Location: $PACK_DIR" -ForegroundColor Yellow

    # List Python tool EXEs
    if ($hasPythonBundle) {
        Write-Host ""
        Write-Host "  Python EXEs (shared bundle):" -ForegroundColor Yellow
        foreach ($name in $PYTHON_TOOL_EXECUTABLES) {
            $path = Join-Path $PACK_DIR $name
            if (Test-Path $path) {
                Write-Host "    - $name" -ForegroundColor Gray
            }
        }
    }

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
