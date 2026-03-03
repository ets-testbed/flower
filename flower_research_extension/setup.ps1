param(
    [string]$PythonExe = "py",
    [string]$VenvName = ".venv"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExtRoot = (Resolve-Path $ScriptDir).Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$VenvDir = Join-Path $ExtRoot $VenvName
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "Detected OS: Windows"
Write-Host "Extension root: $ExtRoot"
Write-Host "Creating virtual environment at $VenvDir"

& $PythonExe -m venv $VenvDir
& $VenvPython -m pip install --upgrade pip setuptools wheel

$FrameworkDir = Join-Path $RepoRoot "framework"
if (Test-Path $FrameworkDir) {
    Write-Host "Installing local Flower framework from $FrameworkDir"
    & $VenvPython -m pip install -e "$FrameworkDir[simulation]"
} else {
    Write-Host "Installing Flower from PyPI"
    & $VenvPython -m pip install "flwr[simulation]>=1.5.0"
}

& $VenvPython -m pip install -r (Join-Path $ExtRoot "requirements.txt")
& $VenvPython -m pip install -e $ExtRoot

Write-Host "Setup complete."
Write-Host "Activate with: $VenvDir\Scripts\Activate.ps1"
Write-Host "Run with: python -m flower_research_extension.experiments.run_experiment"
