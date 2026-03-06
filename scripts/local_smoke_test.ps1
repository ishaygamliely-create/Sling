# scripts/local_smoke_test.ps1
# Automated end-to-end smoke test for Sling local dev.
# Compatible with Windows PowerShell 5.1+
#
# Usage (from repo root, services running):
#   .\scripts\local_smoke_test.ps1
#   .\scripts\local_smoke_test.ps1 -ApiBase "http://127.0.0.1:8000" -PollInterval 5
#
# Exit codes:
#   0 = job reached status "done"
#   1 = job reached status "failed"
#   2 = timeout or connection error

param(
    [string]$ApiBase = "http://127.0.0.1:8000",
    [string]$VideoUrl = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    [int]$PollInterval = 5,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"

function Write-Step { param([string]$msg); Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok { param([string]$msg); Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg); Write-Host "  !   $msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$msg); Write-Host "  ERR $msg" -ForegroundColor Red }

function Coalesce {
    param($value, $fallback)
    if ($null -eq $value -or $value -eq "") { return $fallback }
    return $value
}

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "  Sling Local Smoke Test" -ForegroundColor Cyan
Write-Host "  API : $ApiBase" -ForegroundColor Cyan
Write-Host "  URL : $VideoUrl" -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""

# -- Step 1: API health --------------------------------------------------------
Write-Step "1/4  Checking API health..."
try {
    $health = Invoke-RestMethod "$ApiBase/health"
    Write-Ok "API up -- pipeline=$($health.pipeline)"
}
catch {
    Write-Fail "Cannot reach API at $ApiBase -- is it running?"
    exit 2
}

# -- Step 2: Submit job --------------------------------------------------------
Write-Step "2/4  Submitting job..."
try {
    $body = '{"video_url":"' + $VideoUrl + '"}'
    $r = Invoke-RestMethod "$ApiBase/analyze" -Method POST -ContentType "application/json" -Body $body
    $jobId = $r.job_id
    Write-Ok "Accepted -- job_id: $jobId"
}
catch {
    Write-Fail "POST /analyze failed: $_"
    exit 2
}

# -- Step 3: Poll until terminal state -----------------------------------------
Write-Step "3/4  Polling (timeout=${TimeoutSeconds}s)..."
$start = Get-Date
$status = "queued"
$last = ""

do {
    Start-Sleep -Seconds $PollInterval
    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    try {
        $job = Invoke-RestMethod "$ApiBase/jobs/$jobId"
    }
    catch {
        Write-Fail "GET /jobs/$jobId failed: $_"
        exit 2
    }

    $status = $job.status
    $frames = $job.progress.frames_processed
    $line = "  [${elapsed}s] $status | frames=$frames"

    if ($line -ne $last) {
        switch ($status) {
            "queued" { Write-Host $line -ForegroundColor Gray }
            "running" { Write-Host $line -ForegroundColor Yellow }
            "done" { Write-Host $line -ForegroundColor Green }
            "failed" { Write-Host $line -ForegroundColor Red }
            default { Write-Host $line }
        }
        $last = $line
    }

    if ($elapsed -ge $TimeoutSeconds) {
        Write-Fail "Timed out after ${TimeoutSeconds}s -- job still '$status'"
        exit 2
    }

} while ($status -eq "queued" -or $status -eq "running")

# -- Step 4: Result ------------------------------------------------------------
Write-Host ""
Write-Step "4/4  Result:"
Write-Host ""

if ($status -eq "done") {
    $res = $job.result

    $fHome = Coalesce $res.formation_home  "(null - no football detected)"
    $fAway = Coalesce $res.formation_away  "(null - no football detected)"
    $pHome = Coalesce $res.avg_pressing_height_home "n/a"
    $pAway = Coalesce $res.avg_pressing_height_away "n/a"
    $ratio = Coalesce $res.both_settled_ratio "0"
    $frames = Coalesce $res.frames_processed "0"

    Write-Host "  --------------------------------------------------------" -ForegroundColor Green
    Write-Host "  status           : done" -ForegroundColor Green
    Write-Host "  frames_processed : $frames" -ForegroundColor Green
    Write-Host "  formation_home   : $fHome" -ForegroundColor Green
    Write-Host "  formation_away   : $fAway" -ForegroundColor Green
    Write-Host "  press_home       : $pHome" -ForegroundColor Green
    Write-Host "  press_away       : $pAway" -ForegroundColor Green
    Write-Host "  settled_ratio    : $ratio" -ForegroundColor Green
    Write-Host "  --------------------------------------------------------" -ForegroundColor Green
    Write-Host ""
    Write-Ok "PASS -- full pipeline ran successfully."
    if ($null -eq $res.formation_home) {
        Write-Warn "Formations are null -- expected for non-football video."
        Write-Warn "Use a real broadcast clip for tactical validation."
    }
    exit 0
}
else {
    $errMsg = Coalesce $job.error "(no error message)"
    Write-Host "  --------------------------------------------------------" -ForegroundColor Red
    Write-Host "  status : $status" -ForegroundColor Red
    Write-Host "  error  : $errMsg" -ForegroundColor Red
    Write-Host "  --------------------------------------------------------" -ForegroundColor Red
    Write-Host ""
    Write-Fail "FAIL -- job ended with status '$status'."
    exit 1
}
