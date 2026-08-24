<#
    Run SentenAI locally: starts the FastAPI backend (:8000) and the Next.js
    frontend (:3000), each in its own console window, for demo/dev use on a
    single laptop (mirrors the `make api` / `make web` targets in Makefile).

    Usage:
      .\run.ps1                 Start API + Web (assumes deps already installed)
      .\run.ps1 -Install        Run `uv sync` + `npm install` first
      .\run.ps1 -Seed           Reset + reseed the demo database before starting
      .\run.ps1 -NoBrowser      Don't auto-open the browser once ready
      .\run.ps1 -Stop           Stop whatever is listening on :3000 / :8000
      .\run.ps1 -Force          Stop whatever's on :3000 / :8000 first, then start

      .\run.ps1 -Docker              Build + run the full containerized stack
                                      (Postgres + API + Web, see infra/docker-compose.yml)
      .\run.ps1 -Docker -Seed        ...and seed the demo shop/users into Postgres
      .\run.ps1 -Docker -Stop        Stop and remove the containerized stack
      .\run.ps1 -Docker -Force       Stop the containerized stack first, then rebuild + start
#>

param(
    [switch]$Install,
    [switch]$Seed,
    [switch]$NoBrowser,
    [switch]$Stop,
    [switch]$Force,
    [switch]$Docker
)

$ErrorActionPreference = "Stop"
$root       = Split-Path -Parent $MyInvocation.MyCommand.Path
$apiDir     = Join-Path $root "apps\api"
$webDir     = Join-Path $root "apps\web"
$composeFile = Join-Path $root "infra\docker-compose.yml"
$dockerEnvFile    = Join-Path $root "infra\.env"
$dockerEnvExample = Join-Path $root "infra\.env.example"

function Get-PortOwnerPid([int]$port) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) { return $conn.OwningProcess }
    return $null
}

# Recursively stops a process and all of its descendants (children first, so nothing
# gets re-parented/orphaned mid-walk). uvicorn's --reload mode re-execs itself as a
# grandchild python.exe whose own CommandLine doesn't necessarily contain "uvicorn" at
# all (its reload bootstrap doesn't preserve the original argv) - killing just the
# matched launcher process leaves that grandchild running and still holding the port.
function Stop-ProcessTree([int]$processId) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$processId" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-ProcessTree $_.ProcessId }
    try { Stop-Process -Id $processId -Force -ErrorAction Stop } catch {}
}

# `uv run` / `npm run dev` spawn the real server as a *child* process; killing just the
# PID Get-NetTCPConnection reports (sometimes the launcher, not the child actually bound
# to the socket) can leave that child running as an orphan that still holds the port.
# This finds the launcher by command line, scoped to $root so it can never touch an
# unrelated process elsewhere on the machine, then kills its whole descendant tree.
function Stop-ByCommandLine([string]$pattern) {
    $escapedRoot = [regex]::Escape($root)
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $escapedRoot -and $_.CommandLine -match $pattern } |
        ForEach-Object {
            Write-Host "  Stopping orphaned process tree (PID $($_.ProcessId)): $($_.CommandLine)" -ForegroundColor DarkYellow
            Stop-ProcessTree $_.ProcessId
        }
}

function Stop-Port([int]$port, [string]$label, [string]$cmdlinePattern) {
    $procId = Get-PortOwnerPid $port
    if (-not $procId) {
        Write-Host "$label (port $port) is not running." -ForegroundColor DarkGray
        return
    }

    Write-Host "Stopping $label (PID $procId) on port $port..." -ForegroundColor Yellow
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue

    # Poll for the port to actually clear before trying harder.
    for ($i = 0; $i -lt 6; $i++) {
        Start-Sleep -Milliseconds 500
        if (-not (Get-PortOwnerPid $port)) {
            Write-Host "$label stopped." -ForegroundColor Green
            return
        }
    }

    $stillId = Get-PortOwnerPid $port
    if ($stillId) {
        # Redirecting a native exe's stderr (2>&1) under $ErrorActionPreference = "Stop"
        # turns even a harmless "process already gone" message into a script-terminating
        # error in PS 5.1 - wrap it so the *actual* success check below (Get-PortOwnerPid)
        # is what decides the outcome, not taskkill's own exit status.
        try { taskkill /PID $stillId /F 2>&1 | Out-Null } catch {}
        Start-Sleep -Milliseconds 500
    }

    if (Get-PortOwnerPid $port) {
        # Last resort: the reported PID may belong to a launcher whose real server
        # child is still alive and holding the socket. Find and stop it by command line.
        if ($cmdlinePattern) {
            Stop-ByCommandLine $cmdlinePattern
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not (Get-PortOwnerPid $port)) {
        Write-Host "$label stopped." -ForegroundColor Green
    } else {
        Write-Warning "Port $port is still held after trying to stop it. Close its console window manually, or open a fresh shell and retry."
    }
}

function Wait-ForUrl([string]$url, [int]$timeoutSec, [string]$label) {
    Write-Host "Waiting for $label ($url)..." -ForegroundColor DarkGray
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -eq 200) {
                Write-Host "$label is up." -ForegroundColor Green
                return $true
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    Write-Warning "$label did not respond within $timeoutSec s. Check its console window for errors."
    return $false
}

# --- Docker mode: build + run the full containerized stack (see infra/docker-compose.yml) ---
if ($Docker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "docker not found on PATH. Install Docker Desktop first: https://www.docker.com/products/docker-desktop/"
    }

    if ($Stop -or $Force) {
        # `docker compose down` still needs to interpolate the whole compose file (including
        # the *required* JWT_SECRET var) even though stopping never uses its value - so a
        # missing infra/.env would otherwise make it impossible to ever stop the stack.
        # A process-local placeholder is enough; it's never used to sign anything here.
        if (-not $env:JWT_SECRET -and -not (Test-Path $dockerEnvFile)) {
            $env:JWT_SECRET = "stop-only-placeholder-not-used-to-sign-anything"
        }
    }

    if ($Stop) {
        Write-Host "Stopping containerized stack..." -ForegroundColor Yellow
        docker compose -f $composeFile down
        exit 0
    }

    if ($Force) {
        Write-Host "Stopping containerized stack first (-Force)..." -ForegroundColor Yellow
        docker compose -f $composeFile down
    }

    if (-not (Test-Path $dockerEnvFile)) {
        Write-Host "infra/.env not found - generating one with a fresh random JWT_SECRET..." -ForegroundColor Cyan
        # RNGCryptoServiceProvider (not RandomNumberGenerator.Fill, a .NET 6+-only API) so
        # this works under Windows PowerShell 5.1 / .NET Framework, not just PS 7+.
        $rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::new()
        $bytes = New-Object byte[] 32
        $rng.GetBytes($bytes)
        $rng.Dispose()
        $secret = -join ($bytes | ForEach-Object { $_.ToString("x2") })
        Set-Content -Path $dockerEnvFile -Encoding utf8 -NoNewline -Value "JWT_SECRET=$secret`nENVIRONMENT=production`n"
        Write-Host "Created infra/.env (see infra/.env.example for what these settings mean)." -ForegroundColor DarkGray
    }

    Write-Host "Building + starting Postgres + API + Web..." -ForegroundColor Cyan
    Write-Host "(first run can take several minutes - the API image installs PyTorch)" -ForegroundColor DarkGray
    docker compose -f $composeFile up -d --build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker compose up failed - see output above (often a healthcheck failing to become ready in time; 'docker compose -f infra/docker-compose.yml logs' has the detail)."
    }

    # The ABSA model (~500MB) cold-loads lazily on the API's first request, so give
    # it real headroom instead of the usual few-second web-server startup budget.
    Wait-ForUrl "http://localhost:8000/health" 120 "API" | Out-Null
    Wait-ForUrl "http://localhost:3000/login"  60  "Web" | Out-Null

    if ($Seed) {
        Write-Host "Seeding demo database..." -ForegroundColor Cyan
        docker compose -f $composeFile run --rm api python -m app.db.seed
    }

    if (-not $NoBrowser) {
        Start-Process "http://localhost:3000"
    }

    Write-Host ""
    Write-Host "SentenAI (containerized) is running:" -ForegroundColor Green
    Write-Host "  API      : http://localhost:8000"
    Write-Host "  Web      : http://localhost:3000"
    Write-Host "  Postgres : localhost:5432 (sentinel/sentinel)"
    Write-Host ""
    Write-Host "Logs:  docker compose -f infra/docker-compose.yml logs -f" -ForegroundColor DarkGray
    Write-Host "Stop:  .\run.ps1 -Docker -Stop" -ForegroundColor DarkGray
    exit 0
}

if ($Stop) {
    Stop-Port 8000 "API" "uvicorn"
    Stop-Port 3000 "Web" "next-server|next dev"
    exit 0
}

# --- sanity checks ---
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found on PATH. Install it first: https://docs.astral.sh/uv/"
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "npm not found on PATH. Install Node.js first."
}

if ($Install) {
    Write-Host "Installing Python workspace (uv sync)..." -ForegroundColor Cyan
    Push-Location $root
    uv sync
    Pop-Location

    Write-Host "Installing web dependencies (npm install)..." -ForegroundColor Cyan
    Push-Location $webDir
    npm install
    Pop-Location
}

if (-not (Test-Path (Join-Path $webDir "node_modules"))) {
    Write-Warning "apps/web/node_modules not found. Run '.\run.ps1 -Install' first (or 'npm install' inside apps/web)."
    exit 1
}

if ($Force) {
    Stop-Port 8000 "API" "uvicorn"
    Stop-Port 3000 "Web" "next-server|next dev"
}

foreach ($p in 8000, 3000) {
    $procId = Get-PortOwnerPid $p
    if ($procId) {
        Write-Warning "Port $p is already in use (PID $procId). Run '.\run.ps1 -Stop' first, or '.\run.ps1 -Force' to stop it and start in one go."
        exit 1
    }
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if ($Seed) {
    Write-Host "Seeding demo database..." -ForegroundColor Cyan
    Push-Location $root
    if (Test-Path $venvPython) {
        & $venvPython -m app.db.seed
    } else {
        uv run python -m app.db.seed
    }
    Pop-Location
}

Write-Host "Starting API on http://localhost:8000 ..." -ForegroundColor Cyan
$apiCommand = if (Test-Path $venvPython) { "..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000" } else { "uv run uvicorn app.main:app --port 8000" }
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$apiDir'; $apiCommand"
) | Out-Null

Write-Host "Starting Web on http://localhost:3000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$webDir'; npm run dev"
) | Out-Null

# The ABSA model (~500MB) cold-loads lazily on the API's first request, so give
# it real headroom instead of the usual few-second web-server startup budget.
Wait-ForUrl "http://localhost:8000/health" 90 "API"  | Out-Null
Wait-ForUrl "http://localhost:3000"        60 "Web"  | Out-Null

if (-not $NoBrowser) {
    Start-Process "http://localhost:3000"
}

Write-Host ""
Write-Host "SentenAI is running:" -ForegroundColor Green
Write-Host "  API : http://localhost:8000"
Write-Host "  Web : http://localhost:3000"
Write-Host "  Login: admin@demo.com / admin123"
Write-Host ""
Write-Host "Each service runs in its own window. Close those windows, or run '.\run.ps1 -Stop', to stop." -ForegroundColor DarkGray
