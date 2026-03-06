# scripts/local_smoke_test.ps1
# Automated end-to-end smoke test for Sling local dev.
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
    [string]$ApiBase      = "http://127.0.0.1:8000",
    [string]$VideoUrl     = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    [int]$PollInterval    = 5,
    [int]$TimeoutSeconds  = 300
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) {
    Write-Host "  $msg" -ForegroundColor Cyan
}
function Write-Ok([string]$msg) {
    Write-Host "  ✓ $msg" -ForegroundColor Green
}
function Write-Warn([string]$msg) {
    Write-Host "  ! $msg" -ForegroundColor Yellow
}
function Write-Fail([string]$msg) {
    Write-Host "  ✗ $msg" -ForegroundColor Red
}

Write-Host ""
Write-Host "  ══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Sling Local Smoke Test" -ForegroundColor Cyan
Write-Host "  API : $ApiBase" -ForegroundColor Cyan
Write-Host "  URL : $VideoUrl" -ForegroundColor Cyan
Write-Host "  ══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: API health ─────────────────────────────────────────────────────────
Write-Step "1/4  Checking API health..."
try {
    $health = Invoke-RestMethod "$ApiBase/health"
    Write-Ok "API ok — pipeline=$($health.pipeline)"
} catch {
    Write-Fail "Cannot reach API at $ApiBase — is it running?"
    exit 2
}

# ── Step 2: Submit job ─────────────────────────────────────────────────────────
Write-Step "2/4  Submitting job..."
try {
    $body = @{ video_url = $VideoUrl } | ConvertTo-Json
    $r    = Invoke-RestMethod "$ApiBase/analyze" -Method POST -ContentType "application/json" -Body $body
    $jobId = $r.job_id
    Write-Ok "Accepted — job_id: $jobId"
} catch {
    Write-Fail "POST /analyze failed: $_"
    exit 2
}

# ── Step 3: Poll until terminal state ─────────────────────────────────────────
Write-Step "3/4  Polling (timeout=${TimeoutSeconds}s)..."
$start   = Get-Date
$elapsed = 0
$last    = ""

do {
    Start-Sleep -Seconds $PollInterval
    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    try {
        $job = Invoke-RestMethod "$ApiBase/jobs/$jobId"
    } catch {
        Write-Fail "GET /jobs/$jobId failed: $_"
        exit 2
    }

    $status  = $job.status
    $frames  = $job.progress.frames_processed
    $line    = "  [$elapsed`s] $status | frames=$frames"

    # Only print if something changed
    if ($line -ne $last) {
        switch ($status) {
            "queued"  { Write-Host $line -ForegroundColor Gray }
            "running" { Write-Host $line -ForegroundColor Yellow }
            "done"    { Write-Host $line -ForegroundColor Green }
            "failed"  { Write-Host $line -ForegroundColor Red }
            default   { Write-Host $line }
        }
        $last = $line
    }

    if ($elapsed -ge $TimeoutSeconds) {
        Write-Fail "Timed out after ${TimeoutSeconds}s — job still in state '$status'"
        exit 2
    }

} while ($status -in @("queued", "running"))

# ── Step 4: Print result ───────────────────────────────────────────────────────
Write-Host ""
Write-Step "4/4  Result:"
Write-Host ""

if ($status -eq "done") {
    $res = $job.result
    Write-Host "  ┌─ Job Summary ───────────────────────────────────" -ForegroundColor Green
    Write-Host "  │  status          : done ✓" -ForegroundColor Green
    Write-Host "  │  frames_processed: $($res.frames_processed)" -ForegroundColor Green
    Write-Host "  │  formation_home  : $($res.formation_home  ?? '(null — no football detected)')" -ForegroundColor Green
    Write-Host "  │  formation_away  : $($res.formation_away  ?? '(null — no football detected)')" -ForegroundColor Green
    Write-Host "  │  press_home      : $($res.avg_pressing_height_home ?? 'n/a')" -ForegroundColor Green
    Write-Host "  │  press_away      : $($res.avg_pressing_height_away ?? 'n/a')" -ForegroundColor Green
    Write-Host "  │  settled_ratio   : $($res.both_settled_ratio)" -ForegroundColor Green
    Write-Host "  └────────────────────────────────────────────────" -ForegroundColor Green
    Write-Host ""
    Write-Ok "PASS — full pipeline ran successfully."
    if (-not $res.formation_home) {
        Write-Warn "Formations are null — expected for non-football video."
        Write-Warn "Use a real broadcast football clip for tactical validation."
    }
    exit 0
} else {
    Write-Host "  ┌─ Job Error ─────────────────────────────────────" -ForegroundColor Red
    Write-Host "  │  status : failed" -ForegroundColor Red
    Write-Host "  │  error  : $($job.error)" -ForegroundColor Red
    Write-Host "  └────────────────────────────────────────────────" -ForegroundColor Red
    Write-Host ""
    Write-Fail "FAIL — job ended with status '$status'."
    exit 1
}
