#!/usr/bin/env pwsh

param(
    [string]$OutputDir,
    [string]$WinRARPath,
    [switch]$KeepExpanded
)

$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$project_root = Split-Path -Parent $script_dir

if (-not $OutputDir) {
    $OutputDir = $project_root
}

$OutputDir = if ([IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path (Get-Location) $OutputDir
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$package_name = "ar-$timestamp"
$zip_path = Join-Path $OutputDir "$package_name.zip"

$staging_root = Join-Path ([IO.Path]::GetTempPath()) ("rct-ar-pack-" + [guid]::NewGuid().ToString("N"))
$package_root = Join-Path $staging_root $package_name

$required_relative_paths = @(
    "build-pack/convert_manifold.exe",
    "build-pack/insta_data_extraction_ar.exe",
    "build-pack/insta_time_sync.exe",
    "build-pack/insta_compute_poses.exe",
    "ztools/run-ar.ps1",
    "ztools/readme-ar.md"
)

# Derived from the current import closure of convert_manifold.exe inside build-pack.
$convert_manifold_dependency_dlls = @(
    "abseil_dll.dll",
    "boost_iostreams-vc143-mt-x64-1_88.dll",
    "bz2.dll",
    "gflags.dll",
    "glog.dll",
    "libprotobuf.dll",
    "lz4.dll",
    "liblzma.dll",
    "zlib1.dll",
    "zstd.dll"
) | ForEach-Object { "build-pack/$_" }

function Copy-RelativeFile {
    param([string]$RelativePath)

    $source_path = Join-Path $project_root $RelativePath
    if (-not (Test-Path -LiteralPath $source_path)) {
        throw "Required file not found: $RelativePath"
    }

    $destination_path = Join-Path $package_root $RelativePath
    $destination_dir = Split-Path -Parent $destination_path
    if (-not (Test-Path -LiteralPath $destination_dir)) {
        New-Item -ItemType Directory -Path $destination_dir -Force | Out-Null
    }

    Copy-Item -LiteralPath $source_path -Destination $destination_path -Force
    Write-Host "  Copied: $RelativePath"
}

function Get-WinRARExecutable {
    param([string]$PreferredPath)

    $candidates = @()

    if ($PreferredPath) {
        $candidates += $PreferredPath
    }

    $commands = @(
        (Get-Command Rar.exe -ErrorAction SilentlyContinue),
        (Get-Command WinRAR.exe -ErrorAction SilentlyContinue)
    ) | Where-Object { $_ }
    $candidates += $commands | ForEach-Object { $_.Source }

    $registry_keys = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\WinRAR.exe',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\WinRAR.exe',
        'HKLM:\SOFTWARE\WinRAR',
        'HKCU:\SOFTWARE\WinRAR'
    )

    foreach ($key in $registry_keys) {
        if (-not (Test-Path $key)) {
            continue
        }

        $props = Get-ItemProperty -Path $key
        foreach ($prop_name in @('(default)', 'exe64', 'exe32')) {
            $value = $props.PSObject.Properties[$prop_name].Value
            if ($value) {
                $candidates += [string]$value
            }
        }
    }

    $candidates += @(
        'C:\Program Files\WinRAR\Rar.exe',
        'C:\Program Files\WinRAR\WinRAR.exe',
        'C:\Program Files (x86)\WinRAR\Rar.exe',
        'C:\Program Files (x86)\WinRAR\WinRAR.exe'
    )

    foreach ($candidate in $candidates | Where-Object { $_ } | Select-Object -Unique) {
        if (Test-Path -LiteralPath $candidate) {
            if ((Split-Path -Leaf $candidate) -ieq 'WinRAR.exe') {
                $rar_candidate = Join-Path (Split-Path -Parent $candidate) 'Rar.exe'
                if (Test-Path -LiteralPath $rar_candidate) {
                    return $rar_candidate
                }
            }
            return $candidate
        }
    }

    throw "WinRAR executable not found. Use -WinRARPath to specify WinRAR.exe or Rar.exe explicitly."
}

try {
    $winrar_exe = Get-WinRARExecutable -PreferredPath $WinRARPath

    Write-Host "=== AR Package Builder ===" -ForegroundColor Cyan
    Write-Host "Project Root:  $project_root"
    Write-Host "Output Dir:    $OutputDir"
    Write-Host "Package Name:  $package_name"
    Write-Host "WinRAR:        $winrar_exe"
    Write-Host ""

    if (-not (Test-Path -LiteralPath $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    if (Test-Path -LiteralPath $zip_path) {
        throw "Package already exists: $zip_path"
    }

    New-Item -ItemType Directory -Path $package_root -Force | Out-Null

    Write-Host "Copying required files..." -ForegroundColor Yellow
    foreach ($relative_path in $required_relative_paths) {
        Copy-RelativeFile -RelativePath $relative_path
    }

    Write-Host "Copying convert_manifold dependencies..." -ForegroundColor Yellow
    foreach ($relative_path in $convert_manifold_dependency_dlls) {
        Copy-RelativeFile -RelativePath $relative_path
    }

    Write-Host "Creating zip archive with WinRAR..." -ForegroundColor Yellow
    Push-Location $staging_root
    try {
        & $winrar_exe a -r $zip_path $package_name
        if ($LASTEXITCODE -ne 0) {
            throw "WinRAR failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $zip_path)) {
        throw "WinRAR reported success, but archive was not created: $zip_path"
    }

    $zip_info = Get-Item -LiteralPath $zip_path
    $zip_size_mb = [Math]::Round($zip_info.Length / 1MB, 2)

    Write-Host ""
    Write-Host "[OK] Package created: $zip_path" -ForegroundColor Green
    Write-Host "[OK] Package size: ${zip_size_mb} MB" -ForegroundColor Green

    if ($KeepExpanded) {
        Write-Host "[OK] Expanded package kept at: $package_root" -ForegroundColor Green
    }
}
finally {
    if ((Test-Path -LiteralPath $staging_root) -and -not $KeepExpanded) {
        Remove-Item -LiteralPath $staging_root -Recurse -Force -ErrorAction SilentlyContinue
    }
}