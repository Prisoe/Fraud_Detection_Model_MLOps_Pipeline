param(
    [Parameter(Mandatory=$true)]  [string]$Region,
    [Parameter(Mandatory=$false)] [string]$StackName    = "FraudDetectionStack",
    [Parameter(Mandatory=$false)] [string]$EndpointName = "fraud-detection-endpoint",
    [Parameter(Mandatory=$false)] [string]$InstanceType = "ml.t2.medium",
    [Parameter(Mandatory=$false)] [int]   $InstanceCount = 1,
    [Parameter(Mandatory=$false)] [int]   $CanaryPct    = 10,
    [switch]$Wait
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Reading stack outputs from $StackName ===" -ForegroundColor Cyan
$stackJson = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs" --output json 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stack '$StackName' not found. Run deploy.ps1 first." -ForegroundColor Red
    exit 1
}

$outputs  = $stackJson | ConvertFrom-Json
$ROLE_ARN = ($outputs | Where-Object { $_.OutputKey -eq "SageMakerRoleArn"    }).OutputValue
$BUCKET   = ($outputs | Where-Object { $_.OutputKey -eq "ArtifactsBucketName" }).OutputValue
$SNS_ARN  = ($outputs | Where-Object { $_.OutputKey -eq "AlertsTopicArn"      }).OutputValue

if (-not $ROLE_ARN -or -not $BUCKET) {
    Write-Host "ERROR: Could not read SageMakerRoleArn or ArtifactsBucketName from stack." -ForegroundColor Red
    exit 1
}

Write-Host "  Bucket  : $BUCKET"
Write-Host "  Role    : $ROLE_ARN"
Write-Host "  Canary  : $CanaryPct% to challenger"

$env:AWS_REGION    = $Region
$env:SNS_TOPIC_ARN = $SNS_ARN

if ($Wait) {
    python src/deploy/deploy_endpoint.py --region $Region --role-arn $ROLE_ARN --artifact-bucket $BUCKET --endpoint-name $EndpointName --instance-type $InstanceType --instance-count $InstanceCount --canary-pct $CanaryPct --sns-topic-arn $SNS_ARN --wait
} else {
    python src/deploy/deploy_endpoint.py --region $Region --role-arn $ROLE_ARN --artifact-bucket $BUCKET --endpoint-name $EndpointName --instance-type $InstanceType --instance-count $InstanceCount --canary-pct $CanaryPct --sns-topic-arn $SNS_ARN
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Deployment failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "OK - Endpoint deployment submitted: $EndpointName" -ForegroundColor Green
Write-Host "  https://console.aws.amazon.com/sagemaker/home?region=$Region#/endpoints"
