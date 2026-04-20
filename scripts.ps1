#!/usr/bin/env pwsh

param(
  [Parameter(Mandatory = $true)]
  [string]$ImageDir,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  [ValidateSet("Release", "RelWithDebInfo", "Debug")]
  [string]$Config = "RelWithDebInfo",

  [ValidateSet("sequential", "exhaustive")]
  [string]$PairMode = "sequential",

  [string]$DatabasePath = "",
  [string]$ImageListPath = "",
  [string]$PairListPath = "",
  [string]$BackboneEngine = "",
  [string]$SddhEngine = "",
  [string]$LightGlueEngine = "",
  [string]$TensorRTBinDir = "",
  [string]$VcpkgBinDir = "",
  [int]$MaxEdge = 1024,
  [int]$TopK = 5000,
  [int]$MaxMatches = 4000,
  [switch]$SkipExisting,
  [switch]$OverwriteExisting,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Resolve-OptionalPath {
  param(
    [string]$PathValue,
    [string]$BaseDir
  )

  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    return ""
  }

  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}

function Invoke-NativeStep {
  param(
    [string]$Title,
    [string]$ExePath,
    [string[]]$Arguments
  )

  Write-Host ""
  Write-Host "=== $Title ===" -ForegroundColor Cyan
  Write-Host "$ExePath $($Arguments -join ' ')" -ForegroundColor DarkGray
  & $ExePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Title failed with exit code $LASTEXITCODE"
  }
}

function Get-ImageFiles {
  param([string]$Dir)

  $patterns = @("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
  $items = foreach ($pattern in $patterns) {
    Get-ChildItem -Path $Dir -File -Filter $pattern -ErrorAction SilentlyContinue
  }

  return $items |
    Sort-Object Name |
    Select-Object -Unique
}

function Write-ImageList {
  param(
    [string]$Destination,
    [System.IO.FileInfo[]]$Images
  )

  $Images.Name | Set-Content -Path $Destination -Encoding ascii
}

function Write-PairList {
  param(
    [string]$Destination,
    [string[]]$ImageNames,
    [string]$Mode
  )

  $pairs = New-Object System.Collections.Generic.List[string]
  if ($Mode -eq "sequential") {
    for ($index = 0; $index -lt $ImageNames.Count - 1; ++$index) {
      $pairs.Add("$($ImageNames[$index]) $($ImageNames[$index + 1])")
    }
  } else {
    for ($left = 0; $left -lt $ImageNames.Count; ++$left) {
      for ($right = $left + 1; $right -lt $ImageNames.Count; ++$right) {
        $pairs.Add("$($ImageNames[$left]) $($ImageNames[$right])")
      }
    }
  }

  if ($pairs.Count -eq 0) {
    throw "No valid image pairs generated. Need at least two images."
  }

  $pairs | Set-Content -Path $Destination -Encoding ascii
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sfmRoot = Join-Path $repoRoot "sfm-phoenix"

$resolvedImageDir = Resolve-OptionalPath -PathValue $ImageDir -BaseDir $repoRoot
if (-not (Test-Path $resolvedImageDir)) {
  throw "Image directory not found: $resolvedImageDir"
}

$resolvedOutputDir = Resolve-OptionalPath -PathValue $OutputDir -BaseDir $repoRoot
if (-not (Test-Path $resolvedOutputDir)) {
  New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
}

$resolvedDatabasePath = if ([string]::IsNullOrWhiteSpace($DatabasePath)) {
  Join-Path $resolvedOutputDir "phoenix.db"
} else {
  Resolve-OptionalPath -PathValue $DatabasePath -BaseDir $repoRoot
}

$resolvedImageListPath = if ([string]::IsNullOrWhiteSpace($ImageListPath)) {
  Join-Path $resolvedOutputDir "images.txt"
} else {
  Resolve-OptionalPath -PathValue $ImageListPath -BaseDir $repoRoot
}

$resolvedPairListPath = if ([string]::IsNullOrWhiteSpace($PairListPath)) {
  Join-Path $resolvedOutputDir "pairs.txt"
} else {
  Resolve-OptionalPath -PathValue $PairListPath -BaseDir $repoRoot
}

$resolvedBackboneEngine = if ([string]::IsNullOrWhiteSpace($BackboneEngine)) {
  Join-Path $sfmRoot "engines\aliked_backbone.engine"
} else {
  Resolve-OptionalPath -PathValue $BackboneEngine -BaseDir $repoRoot
}

$resolvedSddhEngine = if ([string]::IsNullOrWhiteSpace($SddhEngine)) {
  Join-Path $sfmRoot "engines\aliked_sddh.engine"
} else {
  Resolve-OptionalPath -PathValue $SddhEngine -BaseDir $repoRoot
}

$resolvedLightGlueEngine = if ([string]::IsNullOrWhiteSpace($LightGlueEngine)) {
  Join-Path $sfmRoot "engines\lightglue.engine"
} else {
  Resolve-OptionalPath -PathValue $LightGlueEngine -BaseDir $repoRoot
}

$binaryDir = Join-Path $sfmRoot "build\$Config"
$extractorExe = Join-Path $binaryDir "phoenix_feature_extractor.exe"
$matcherExe = Join-Path $binaryDir "phoenix_matcher.exe"

foreach ($requiredPath in @(
  $resolvedImageDir,
  $resolvedBackboneEngine,
  $resolvedSddhEngine,
  $resolvedLightGlueEngine,
  $extractorExe,
  $matcherExe
)) {
  if (-not (Test-Path $requiredPath)) {
    throw "Required path not found: $requiredPath"
  }
}

if ($Clean -and (Test-Path $resolvedDatabasePath)) {
  Remove-Item -Path $resolvedDatabasePath -Force
}

$images = Get-ImageFiles -Dir $resolvedImageDir
if ($images.Count -lt 2) {
  throw "Need at least two images under: $resolvedImageDir"
}

if ([string]::IsNullOrWhiteSpace($ImageListPath)) {
  Write-ImageList -Destination $resolvedImageListPath -Images $images
} elseif (-not (Test-Path $resolvedImageListPath)) {
  throw "Image list not found: $resolvedImageListPath"
}

$imageNames = Get-Content -Path $resolvedImageListPath |
  Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

if ($imageNames.Count -lt 2) {
  throw "Image list must contain at least two entries: $resolvedImageListPath"
}

if ([string]::IsNullOrWhiteSpace($PairListPath)) {
  Write-PairList -Destination $resolvedPairListPath -ImageNames $imageNames -Mode $PairMode
} elseif (-not (Test-Path $resolvedPairListPath)) {
  throw "Pair list not found: $resolvedPairListPath"
}

$pathPrefix = New-Object System.Collections.Generic.List[string]
if (-not [string]::IsNullOrWhiteSpace($TensorRTBinDir)) {
  $pathPrefix.Add((Resolve-OptionalPath -PathValue $TensorRTBinDir -BaseDir $repoRoot))
} elseif (Test-Path "C:\Program Files\TensorRT-10.16.1.11\bin") {
  $pathPrefix.Add("C:\Program Files\TensorRT-10.16.1.11\bin")
}

if (-not [string]::IsNullOrWhiteSpace($VcpkgBinDir)) {
  $pathPrefix.Add((Resolve-OptionalPath -PathValue $VcpkgBinDir -BaseDir $repoRoot))
} elseif (Test-Path "F:\Library\vcpkg\installed\x64-windows\bin") {
  $pathPrefix.Add("F:\Library\vcpkg\installed\x64-windows\bin")
}

if ($pathPrefix.Count -gt 0) {
  $env:PATH = ($pathPrefix -join ';') + ';' + $env:PATH
}

$extractorArgs = @(
  "--database_path", $resolvedDatabasePath,
  "--image_path", $resolvedImageDir,
  "--image_list_path", $resolvedImageListPath,
  "--Phoenix.backbone_engine", $resolvedBackboneEngine,
  "--Phoenix.sddh_engine", $resolvedSddhEngine,
  "--Phoenix.max_edge", $MaxEdge,
  "--Phoenix.top_k", $TopK
)

if ($SkipExisting) {
  $extractorArgs += @("--Phoenix.skip_existing", "1")
} else {
  $extractorArgs += @("--Phoenix.skip_existing", "0")
}

$matcherArgs = @(
  "--database_path", $resolvedDatabasePath,
  "--pair_list_path", $resolvedPairListPath,
  "--Phoenix.lightglue_engine", $resolvedLightGlueEngine,
  "--Phoenix.max_edge", $MaxEdge,
  "--Phoenix.max_matches", $MaxMatches
)

if ($SkipExisting) {
  $matcherArgs += @("--Phoenix.skip_existing", "1")
} else {
  $matcherArgs += @("--Phoenix.skip_existing", "0")
}

if ($OverwriteExisting) {
  $matcherArgs += @("--Phoenix.overwrite_existing", "1")
}

Write-Host "=== Phoenix Data Processing ===" -ForegroundColor Cyan
Write-Host "Image Dir:      $resolvedImageDir"
Write-Host "Output Dir:     $resolvedOutputDir"
Write-Host "Database:       $resolvedDatabasePath"
Write-Host "Image List:     $resolvedImageListPath"
Write-Host "Pair List:      $resolvedPairListPath"
Write-Host "Config:         $Config"
Write-Host "Pair Mode:      $PairMode"
Write-Host "Max Edge:       $MaxEdge"
Write-Host "Top K:          $TopK"
Write-Host "Max Matches:    $MaxMatches"
Write-Host "Image Count:    $($imageNames.Count)"
Write-Host ""
Write-Host "Note: migration logger prints readable logs in Debug and RelWithDebInfo, encrypted logs in Release." -ForegroundColor Yellow

Invoke-NativeStep -Title "Phoenix Feature Extraction" -ExePath $extractorExe -Arguments $extractorArgs
Invoke-NativeStep -Title "Phoenix Matching" -ExePath $matcherExe -Arguments $matcherArgs

Write-Host ""
Write-Host "[OK] Processing completed" -ForegroundColor Green
Write-Host "Database:  $resolvedDatabasePath"
Write-Host "Images:    $resolvedImageListPath"
Write-Host "Pairs:     $resolvedPairListPath"