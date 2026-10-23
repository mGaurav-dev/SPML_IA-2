# PowerShell runner script for IA-2 experiments
Param(
    [string]$HouseholdId = "MAC000002",
    [int]$SubsetSize = 25000,
    [int]$Repeats = 5,
    [switch]$SmokeTest,
    [switch]$SkipPlots
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -Path $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  IA-2: Execution Pipeline Runner" -ForegroundColor Cyan
Write-Host "  Working Directory: $ProjectRoot" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$CmdArgs = @("main.py", "--household-id", $HouseholdId, "--subset-size", $SubsetSize, "--n-repeats", $Repeats)

if ($SmokeTest) {
    $CmdArgs += "--smoke-test"
    Write-Host "[INFO] Running in smoke-test mode with synthetic data." -ForegroundColor Yellow
}

if ($SkipPlots) {
    $CmdArgs += "--skip-plots"
}

Write-Host "[EXEC] python $($CmdArgs -join ' ')" -ForegroundColor Green
python @CmdArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Pipeline execution finished successfully." -ForegroundColor Green
} else {
    Write-Host "[ERROR] Pipeline execution exited with code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}
