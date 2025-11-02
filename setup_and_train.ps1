<#
setup_and_train.ps1

Creates/activates a virtual environment, installs requirements, and runs model training.

Usage (PowerShell, run from project root):
  # test run (1 epoch, small batch)
  .\setup_and_train.ps1 -Epochs 1 -BatchSize 16

  # full run (example)
  .\setup_and_train.ps1 -Epochs 25 -BatchSize 64

Notes:
- This script only prepares the environment and invokes Python in your shell. It does not execute any commands on your behalf remotely.
- If you prefer to run commands manually, follow the steps in README.md instead.
#>

param(
    [int]$Epochs = 25,
    [int]$BatchSize = 64,
    [switch]$NoInstall
)

Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $projectRoot

Write-Host "Project root: $projectRoot"

# 1) Create venv if needed
if (-not (Test-Path -Path "$projectRoot\.venv")) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
} else {
    Write-Host "Virtual environment .venv already exists."
}

Write-Host "Activating virtual environment..."
# Correct activation: dot-sourcing the venv activate script
. .\.venv\Scripts\Activate.ps1

if (-not $NoInstall) {
    Write-Host "Upgrading pip, setuptools, wheel..."
    python -m pip install --upgrade pip setuptools wheel

    Write-Host "Installing requirements from requirements.txt..."
    python -m pip install --no-cache-dir -r requirements.txt
} else {
    Write-Host "Skipping dependency installation (--NoInstall specified)."
}

Write-Host "Starting training: epochs=$Epochs, batch_size=$BatchSize"
python model.py --epochs $Epochs --batch-size $BatchSize

Pop-Location
