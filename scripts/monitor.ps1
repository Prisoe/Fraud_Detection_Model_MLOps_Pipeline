<#
.SYNOPSIS
  Run PSI + label drift check for the fraud detection endpoint.

.EXAMPLE
  .\scripts\monitor.ps1 -Region us-east-1
  .\scripts\monitor.ps1 -Region us-east-1 -AutoRetrain -Threshold 0.20
#>
param(
    [Parameter(Mandatory=$true)]  [string]$Region,
    [Parameter(Mandatory=$false)] [string]$StackName    = "FraudDetectionStack",
    [Parameter(Mandatory=$false)] [string]$EndpointName = "fraud-detection-endpoint",
    [Parameter(Mandatory=$false)] [float] $Threshold    = 0.25,
    [switch]$AutoRetrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "`n=== Reading stack outputs from $StackName ===" -ForegroundColor Cyan
$stackJson = aws cloudformation describe-stacks `
    --stack-name $StackName `
    --region $Region `
    --query "Stacks[0].Outputs" `
    --output json 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Stack '$StackName' not found. Run deploy.ps1 first." -ForegroundColor Red
    exit 1
}

$outputs = $stackJson | ConvertFrom-Json
$BUCKET  = ($outputs | Where-Object { $_.OutputKey -eq "ArtifactsBucketName" }).OutputValue
$SNS_ARN = ($outputs | Where-Object { $_.OutputKey -eq "AlertsTopicArn"      }).OutputValue

if (-not $BUCKET) {
    Write-Host "❌ Could not read ArtifactsBucketName from stack outputs." -ForegroundColor Red
    exit 1
}

Write-Host "  Bucket    : $BUCKET"
Write-Host "  Threshold : $Threshold"
Write-Host "  AutoRetrain: $($AutoRetrain.IsPresent)"

# Run drift check
$cmd = @(
    "python", "src/monitoring/model_monitor_setup.py",
    "--region",              $Region,
    "--baseline-s3-uri",     "s3://$BUCKET/data/baseline/baseline.csv",
    "--recent-s3-prefix",    "s3://$BUCKET/monitoring/data-capture/$EndpointName/",
    "--artifact-bucket",     $BUCKET,
    "--psi-threshold",       $Threshold,
    "--sns-topic-arn",       $SNS_ARN
)
if ($AutoRetrain) { $cmd += "--auto-retrain" }

& $cmd[0] $cmd[1..$cmd.Length]
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Drift check failed." -ForegroundColor Red; exit 1 }

# Ensure CloudWatch alarm exists
Write-Host "`n=== Upserting CloudWatch alarm ===" -ForegroundColor Cyan
python src/monitoring/alarms.py `
    --region $Region `
    --sns-topic-arn $SNS_ARN `
    --threshold $Threshold

Write-Host "`n✅ Monitoring complete" -ForegroundColor Green
