# scripts/start_local.ps1
# One-command local startup for Sling dev.
# Usage (from repo root, venv active):
#   .venv\Scripts\Activate.ps1
#   .\scripts\start_local.ps1

param(
    [int]$WorkerPort = 9000,
    [int]$ApiPort    = 8000
)

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

# ── Load .env.local ────────────────────────────────────────────────────────────
$EnvFile = Join-Path $Root ".env.local"
if (Test-Path $EnvFile) {
    Write-Host "  Loading $EnvFile ..." -ForegroundColor Cyan
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $k, $v = $line -split "=", 2
            [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
        }
    }
} else {
    Write-Host "  [warn] No .env.local found — using existing env vars" -ForegroundColor Yellow
    # Defaults for local dev if no .env.local
    if (-not $env:DEV_MODE)          { $env:DEV_MODE          = "1" }
    if (-not $env:WORKER_AUTH_TOKEN) { $env:WORKER_AUTH_TOKEN = "localtest" }
    if (-not $env:WORKER_BASE_URL)   { $env:WORKER_BASE_URL   = "http://127.0.0.1:$WorkerPort" }
}

# ── Start Worker in background ─────────────────────────────────────────────────
Write-Host ""
Write-Host "  Starting Worker on :$WorkerPort ..." -ForegroundColor Green
$WorkerJob = Start-Job -ScriptBlock {
    param($root, $port)
    Set-Location $root
    # Copy env vars into job scope
    $env:DEV_MODE          = $using:env:DEV_MODE
    $env:WORKER_AUTH_TOKEN = $using:env:WORKER_AUTH_TOKEN
    & python run_worker.py
} -ArgumentList $Root, $WorkerPort

# Give Worker a moment to boot
Start-Sleep -Seconds 3

# Quick health check
try {
    $hc = Invoke-RestMethod "http://127.0.0.1:$WorkerPort/health" -ErrorAction Stop
    Write-Host "  Worker health: $($hc.status) | store=$($hc.store) | dev_mode=$($hc.dev_mode)" -ForegroundColor Green
} catch {
    Write-Host "  [warn] Worker health check failed — check logs below if API fails" -ForegroundColor Yellow
}

# ── Print URLs ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ┌─────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "  │  Sling Local Dev                            │" -ForegroundColor Cyan
Write-Host "  │  Worker  → http://127.0.0.1:$WorkerPort         │" -ForegroundColor Cyan
Write-Host "  │  API     → http://127.0.0.1:$ApiPort       (starting...) │" -ForegroundColor Cyan
Write-Host "  │  API Docs→ http://127.0.0.1:$ApiPort/docs   │" -ForegroundColor Cyan
Write-Host "  │  Stop: Ctrl+C (cleans up worker)           │" -ForegroundColor Cyan
Write-Host "  └─────────────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""

# ── Start API in foreground (blocks until Ctrl+C) ──────────────────────────────
try {
    uvicorn api.server:app --port $ApiPort --reload
} finally {
    Write-Host ""
    Write-Host "  Stopping Worker job ..." -ForegroundColor Yellow
    Stop-Job  $WorkerJob -ErrorAction SilentlyContinue
    Remove-Job $WorkerJob -ErrorAction SilentlyContinue
    Write-Host "  Done." -ForegroundColor Green
}
