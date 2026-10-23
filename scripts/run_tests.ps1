# PowerShell test suite runner for IA-2
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -Path $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  IA-2: Automated Pytest Suite Runner" -ForegroundColor Cyan
Write-Host "  Target: $ProjectRoot\tests" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

python -m pytest tests/ -v

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] All unit tests passed successfully." -ForegroundColor Green
} else {
    Write-Host "[ERROR] Test suite failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}
