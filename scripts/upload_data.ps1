[CmdletBinding()]
param(
  [Parameter(Mandatory=$false)]
  [string]$Region = "us-east-1",

  [Parameter(Mandatory=$true)]
  [string]$Bucket
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LocalFile = Join-Path $RepoRoot "ml\sample_data.csv"

if (-not (Test-Path $LocalFile)) {
  throw "sample_data.csv not found at: $LocalFile"
}

Write-Host "Uploading $LocalFile to s3://$Bucket/data/raw/sample_data.csv"
aws s3 cp "$LocalFile" "s3://$Bucket/data/raw/sample_data.csv" --region $Region | Out-Host