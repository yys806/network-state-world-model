param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Command,
    [double]$MaxMemoryPercent = 80.0,
    [int]$CheckIntervalSeconds = 5,
    [string]$WorkDir = (Get-Location).Path
)

if ($Command.Count -lt 1) {
    throw "Command is required."
}
if ($MaxMemoryPercent -le 0 -or $MaxMemoryPercent -gt 100) {
    throw "MaxMemoryPercent must be in (0, 100]."
}
if ($CheckIntervalSeconds -lt 1) {
    throw "CheckIntervalSeconds must be >= 1."
}

function Get-MemoryUsedPercent {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round((1.0 - ($os.FreePhysicalMemory / $os.TotalVisibleMemorySize)) * 100.0, 2)
}

function Stop-ProcessTree {
    param([int]$RootProcessId)
    $children = Get-CimInstance Win32_Process |
        Where-Object { $_.ParentProcessId -eq $RootProcessId }
    foreach ($child in $children) {
        Stop-ProcessTree -RootProcessId ([int]$child.ProcessId)
    }
    Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

$exe = $Command[0]
$arguments = @()
if ($Command.Count -gt 1) {
    $arguments = $Command[1..($Command.Count - 1)]
}

$startMemory = Get-MemoryUsedPercent
Write-Host "[memguard] start memory=${startMemory}% limit=${MaxMemoryPercent}% command=$($Command -join ' ')"
if ($startMemory -ge $MaxMemoryPercent) {
    throw "Memory usage is already at or above limit: ${startMemory}% >= ${MaxMemoryPercent}%."
}

$process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $WorkDir -NoNewWindow -PassThru
try {
    while (-not $process.HasExited) {
        Start-Sleep -Seconds $CheckIntervalSeconds
        $process.Refresh()
        $used = Get-MemoryUsedPercent
        Write-Host "[memguard] pid=$($process.Id) memory=${used}%"
        if ($used -ge $MaxMemoryPercent) {
            Write-Host "[memguard] memory limit exceeded; stopping process tree pid=$($process.Id)"
            Stop-ProcessTree -RootProcessId $process.Id
            throw "Memory usage exceeded limit: ${used}% >= ${MaxMemoryPercent}%."
        }
    }
}
finally {
    $endMemory = Get-MemoryUsedPercent
    Write-Host "[memguard] end memory=${endMemory}%"
}

exit $process.ExitCode
