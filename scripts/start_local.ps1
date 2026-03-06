# scripts/start_local.ps1
# One-command local startup for Sling dev.
# Usage (from repo root, venv active):
#   .venv\Scripts\Activate.ps1
#   .\scripts\start_local.ps1

param(
    [int]$WorkerPort = 9000,
    [int]$ApiPort = 8000
)

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

# -- Load .env.local -----------------------------------------------------------
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
}
else {
    Write-Host "  [warn] No .env.local found -- using defaults" -ForegroundColor Yellow
    if (-not $env:DEV_MODE) { $env:DEV_MODE = "1" }
    if (-not $env:WORKER_AUTH_TOKEN) { $env:WORKER_AUTH_TOKEN = "localtest" }
    if (-not $env:WORKER_BASE_URL) { $env:WORKER_BASE_URL = "http://127.0.0.1:$WorkerPort" }
}

# -- Start Worker in background ------------------------------------------------
Write-Host ""
Write-Host "  Starting Worker on port $WorkerPort ..." -ForegroundColor Green

$workerEnv = @{
    DEV_MODE          = $env:DEV_MODE
    WORKER_AUTH_TOKEN = $env:WORKER_AUTH_TOKEN
}

$WorkerJob = Start-Job -ScriptBlock {
    param($root, $envVars)
    Set-Location $root
    foreach ($kv in $envVars.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, "Process")
    }
    python run_worker.py
} -ArgumentList $Root, $workerEnv

# Give Worker time to boot
Start-Sleep -Seconds 3

# Quick health check (unauthenticated)
try {
    $hc = Invoke-RestMethod "http://127.0.0.1:$WorkerPort/health" -ErrorAction Stop
    Write-Host "  Worker health OK  store=$($hc.store)  dev_mode=$($hc.dev_mode)" -ForegroundColor Green
}
catch {
    Write-Host "  [warn] Worker health check failed -- it may still be starting up" -ForegroundColor Yellow
}

# -- Print URLs ----------------------------------------------------------------
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "  Sling Local Dev" -ForegroundColor Cyan
Write-Host "  Worker  : http://127.0.0.1:$WorkerPort/health" -ForegroundColor Cyan
Write-Host "  API     : http://127.0.0.1:$ApiPort  (starting below...)" -ForegroundColor Cyan
Write-Host "  API docs: http://127.0.0.1:$ApiPort/docs" -ForegroundColor Cyan
Write-Host "  Stop    : Ctrl+C  (worker job cleaned up automatically)" -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""

# -- Start API in foreground (blocks until Ctrl+C) -----------------------------
try {
    uvicorn api.server:app --port $ApiPort --reload
}
finally {
    Write-Host ""
    Write-Host "  Stopping Worker job ..." -ForegroundColor Yellow
    Stop-Job   $WorkerJob -ErrorAction SilentlyContinue
    Remove-Job $WorkerJob -ErrorAction SilentlyContinue
    Write-Host "  Done." -ForegroundColor Green
}
