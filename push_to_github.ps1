<#
push_to_github.ps1

Safe helper script to stage, commit and push this repo to GitHub.

USAGE (PowerShell):
  # preview what will be committed
  .\push_to_github.ps1 -WhatIf

  # real run (will prompt for commit message)
  .\push_to_github.ps1

Notes:
- This script does not embed credentials. Use a Git credential helper, SSH, or GITHUB_TOKEN env var.
- If you want to use a Personal Access Token (PAT) over HTTPS, set the env var GITHUB_PAT and the script will use it to set the remote URL temporarily.
#>

param(
    [string]$RemoteUrl = 'https://github.com/treasurebby/Mood-detector-app.git',
    [string]$Branch = 'main',
    [switch]$UsePAT
)

Set-StrictMode -Version Latest

Write-Host "Checking git status..."
git status

if ($PSCmdlet.MyInvocation.BoundParameters.ContainsKey('WhatIf')) {
    Write-Host "WhatIf: exiting after status"
    return
}

$msg = Read-Host "Enter commit message (or leave blank to use default)"
if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Prepare repo for push (update .gitignore/.gitattributes)" }

Write-Host "Staging all changes..."
git add -A
git commit -m "$msg" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No changes to commit or commit failed. Continuing..."
}

if ($UsePAT) {
    if (-not $env:GITHUB_PAT) {
        Write-Host "GITHUB_PAT environment variable not set. Aborting."; exit 1
    }
    # set remote with embedded token (temporary); this exposes token in git config; user can delete remote after push
    $urlWithToken = $RemoteUrl -replace 'https://', "https://$($env:GITHUB_PAT)@"
    git remote remove origin 2>$null
    git remote add origin $urlWithToken
} else {
    if (-not (git remote get-url origin 2>$null)) {
        git remote add origin $RemoteUrl 2>$null
    }
}

Write-Host "Pushing to $RemoteUrl ($Branch)..."
git branch -M $Branch
git push -u origin $Branch

Write-Host "Push complete. If you used UsePAT, consider removing the remote or clearing credentials." 
