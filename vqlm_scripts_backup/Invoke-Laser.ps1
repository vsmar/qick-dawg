<#
.SYNOPSIS
    Controls the NV lab laser via the QICK RFSoC.

.DESCRIPTION
    Attempts to connect to the Pyro4 daemon on the RFSoC and toggle the laser.
    If the daemon is not running, automatically starts the nameserver and daemon
    over SSH before retrying. Follows PowerShell verb-noun naming conventions.

.PARAMETER State
    The desired laser state. Must be 'On' or 'Off'.

.EXAMPLE
    .\Invoke-Laser.ps1 -State On
    .\Invoke-Laser.ps1 -State Off

.NOTES
    Requires SSH key auth to be configured for the RFSoC (no password prompts).
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("On", "Off")]
    [string]$State
)

# --- Configuration ---
$NS_HOST          = "192.168.3.1"
$NS_PORT          = 8888
$RFSOC_USER       = "xilinx"
$LASER_CONTROL_PY = "C:\Users\QT3 User Facility\Documents\qick-dawg\vqlm_scripts\laser_control.py"
$PYTHON           = "C:\Users\QT3 User Facility\Documents\qick-dawg\.venv\Scripts\python.exe"
$NS_TIMEOUT_S     = 10
$DAEMON_TIMEOUT_S = 60

# --- Helpers ---

function Test-NameserverReachable {
    param([int]$TimeoutMs = 1000)
    $tcp = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $tcp.BeginConnect($NS_HOST, $NS_PORT, $null, $null)
        $success = $result.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        return $success -and $tcp.Connected
    } catch {
        return $false
    } finally {
        $tcp.Close()
    }
}

function Test-DaemonReady {
    $result = ssh "$RFSOC_USER@$NS_HOST" "/usr/local/share/pynq-venv/bin/python3 /home/xilinx/jupyter_notebooks/connect/check_daemon.py"
    return $result -eq "ready"
}

function Start-RFSoCStack {
    Write-Host ""
    Write-Host "Starting RFSoC - this may take up to a minute on first boot..." -ForegroundColor Yellow

    # Step 1: nameserver
    $nsCmd = 'nohup bash -c ''PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 /usr/local/share/pynq-venv/bin/pyro4-ns -n 192.168.3.1 -p 8888'' > /tmp/qick_ns.log 2>&1 &'
    ssh "$RFSOC_USER@$NS_HOST" $nsCmd

    Write-Host "  Waiting for nameserver..." -NoNewline
    $deadline = (Get-Date).AddSeconds($NS_TIMEOUT_S)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 1
        if (Test-NameserverReachable) {
            Write-Host " ready." -ForegroundColor Green
            break
        }
    }
    if (-not (Test-NameserverReachable)) {
        Write-Host " FAILED" -ForegroundColor Red
        return $false
    }

    # Step 2: QICK daemon
    $daemonCmd = 'nohup sudo /home/xilinx/jupyter_notebooks/connect/start_daemon.sh > /tmp/qick_daemon.log 2>&1 &'
    ssh "$RFSOC_USER@$NS_HOST" $daemonCmd

    Write-Host "  Waiting for QICK daemon (loading bitfile)..." -NoNewline
    $deadline = (Get-Date).AddSeconds($DAEMON_TIMEOUT_S)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if (Test-DaemonReady) {
            Write-Host " ready." -ForegroundColor Green
            return $true
        }
    }

    Write-Host " FAILED" -ForegroundColor Red
    return $false
}

function Invoke-LaserCommand {
    & $PYTHON "$LASER_CONTROL_PY" $State.ToLower() | Out-Null
    return $LASTEXITCODE
}

# --- Main ---

# Fast path: daemon already running
$exitCode = Invoke-LaserCommand

if ($exitCode -eq 0) {
    Write-Host "Laser $State." -ForegroundColor Green
    exit 0
}

# Daemon unreachable -> try to start stack
if ($exitCode -eq 2) {

    Write-Host "RFSoC daemon not running." -ForegroundColor Yellow

    $started = Start-RFSoCStack

    if (-not $started) {
        Write-Host ""
        Write-Host "Failed to connect to the RFSoC." -ForegroundColor Red
        Write-Host "Please check that the RFSoC is powered on and reachable at $NS_HOST, then try again." -ForegroundColor Red
        exit 1
    }

    # Retry after startup
    $exitCode = Invoke-LaserCommand

    if ($exitCode -eq 0) {
        Write-Host "Laser $State." -ForegroundColor Green
        exit 0
    }
}

# Any other failure
Write-Host "Failed to set laser $State." -ForegroundColor Red
exit 1