<#
.SYNOPSIS
  Register (or remove) a Windows Scheduled Task that rebuilds the commodity GEX
  dashboard every weekday.

.DESCRIPTION
  Creates a task that runs `run_gex.py` and writes output\gex_dashboard.html.
  Runs only when you are logged on, under your own account, with no stored
  password and no elevation.

  Default time is 17:15 local, which is after the 16:00 ET US equity close for
  Eastern-time users. Open interest is an end-of-day figure, so running before
  the close pairs stale OI with a fresh spot price. Adjust -Time if you are not
  on Eastern time.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File setup_schedule.ps1
  powershell -ExecutionPolicy Bypass -File setup_schedule.ps1 -Time 18:30
  powershell -ExecutionPolicy Bypass -File setup_schedule.ps1 -WhatIfOnly
  powershell -ExecutionPolicy Bypass -File setup_schedule.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$Time = "17:15",
    [string]$TaskName = "CommodityGEX-Daily",
    [string]$Watchlist = "",
    [switch]$Remove,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $here "run_gex.py"

if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        Write-Host "No task named '$TaskName' is registered. Nothing to remove."
        return
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
    return
}

if (-not (Test-Path $runner)) { throw "run_gex.py not found next to this script ($here)." }

# Resolve the launcher. `py` is a shim, so record the real interpreter path -
# Task Scheduler does not resolve PATH the way an interactive shell does.
$python = $null
try { $python = (Get-Command py -ErrorAction Stop).Source } catch {}
if ($null -eq $python) {
    try { $python = (Get-Command python -ErrorAction Stop).Source } catch {}
}
if ($null -eq $python) { throw "Could not find 'py' or 'python' on PATH." }

$argList = "`"$runner`""
if ($Watchlist -ne "") { $argList += " --watchlist $Watchlist" }

Write-Host "Task name  : $TaskName"
Write-Host "Runs       : weekdays at $Time (local time)"
Write-Host "Command    : $python $argList"
Write-Host "Working dir: $here"
Write-Host "Output     : $(Join-Path $here 'output\gex_dashboard.html')"
Write-Host ""

if ($WhatIfOnly) {
    Write-Host "-WhatIfOnly given: nothing was registered."
    return
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    Write-Host "A task named '$TaskName' already exists - replacing it."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute $python -Argument $argList -WorkingDirectory $here
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
# Interactive token: runs as you, when you are logged on, no stored password.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Rebuilds the commodity futures GEX dashboard each weekday." | Out-Null

Write-Host "Registered."
Write-Host ""
Write-Host "Run it now to confirm :  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Check when it last ran:  Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host "Remove it             :  powershell -File setup_schedule.ps1 -Remove"
