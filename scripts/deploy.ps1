param(
    [Parameter(Mandatory=$true)]  [string]$Region,
    [Parameter(Mandatory=$true)]  [string]$EmailForAlerts,
    [Parameter(Mandatory=$false)] [string]$AlertsMode   = "failures",
    [Parameter(Mandatory=$false)] [string]$StackName    = "FraudDetectionStack",
    [Parameter(Mandatory=$false)] [string]$DataPath     = "ml\creditcard.csv",
    [Parameter(Mandatory=$false)] [string]$EndpointName = "fraud-detection-endpoint",
    [Parameter(Mandatory=$false)] [string]$InstanceType = "ml.t2.medium",
    [Parameter(Mandatory=$false)] [switch]$SkipCDK,
    [Parameter(Mandatory=$false)] [switch]$SkipPipeline,
    [Parameter(Mandatory=$false)] [switch]$SkipEndpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-LastCommand([string]$msg) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $msg (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- 0. Pre-flight
Write-Host ""
Write-Host "=== [0/6] Pre-flight checks ===" -ForegroundColor Cyan
$identity = aws sts get-caller-identity 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: AWS credentials not configured. Run: aws configure" -ForegroundColor Red
    exit 1
}
$callerArn = ($identity | ConvertFrom-Json).Arn
Write-Host "  OK - AWS identity: $callerArn"
Write-Host "  OK - Node.js: $(node --version 2>&1)"
Write-Host "  OK - Python: $(python --version 2>&1)"

if (-not $SkipPipeline) {
    if (-not (Test-Path $DataPath)) {
        Write-Host "ERROR: Dataset not found at '$DataPath'" -ForegroundColor Red
        exit 1
    }
    $sizeMB = [math]::Round((Get-Item $DataPath).Length / 1MB, 1)
    Write-Host "  OK - Dataset: $DataPath ($sizeMB MB)"
}

# --- 1. CDK
if (-not $SkipCDK) {
    Write-Host ""
    Write-Host "=== [1/6] CDK bootstrap + deploy ===" -ForegroundColor Cyan
    Push-Location infra
    npm install --silent
    Assert-LastCommand "npm install"
    $account = (aws sts get-caller-identity --query Account --output text)
    npx cdk bootstrap "aws://$account/$Region" --region $Region
    Assert-LastCommand "cdk bootstrap"
    npx cdk deploy $StackName --require-approval never --parameters EmailForAlerts=$EmailForAlerts --parameters AlertsMode=$AlertsMode --region $Region
    Assert-LastCommand "cdk deploy"
    Pop-Location
    Write-Host "  OK - CDK deploy complete"
} else {
    Write-Host ""
    Write-Host "=== [1/6] Skipping CDK ===" -ForegroundColor Yellow
}

# --- 2. Stack outputs
Write-Host ""
Write-Host "=== [2/6] Reading stack outputs ===" -ForegroundColor Cyan
$stackJson = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs" --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Stack '$StackName' not found in $Region." -ForegroundColor Red
    exit 1
}

$outputs  = $stackJson | ConvertFrom-Json
$BUCKET   = ($outputs | Where-Object { $_.OutputKey -eq "ArtifactsBucketName" }).OutputValue
$ROLE_ARN = ($outputs | Where-Object { $_.OutputKey -eq "SageMakerRoleArn"    }).OutputValue
$SNS_ARN  = ($outputs | Where-Object { $_.OutputKey -eq "AlertsTopicArn"      }).OutputValue

if (-not $BUCKET -or -not $ROLE_ARN) {
    Write-Host "ERROR: Could not read stack outputs." -ForegroundColor Red
    exit 1
}

Write-Host "  OK - Bucket  : $BUCKET"
Write-Host "  OK - Role ARN: $ROLE_ARN"
Write-Host "  OK - SNS ARN : $SNS_ARN"

$env:AWS_REGION         = $Region
$env:AWS_DEFAULT_REGION = $Region
$env:SAGEMAKER_ROLE_ARN = $ROLE_ARN
$env:ARTIFACT_BUCKET    = $BUCKET
$env:SNS_TOPIC_ARN      = $SNS_ARN

# --- 3-5. Pipeline
if (-not $SkipPipeline) {
    Write-Host ""
    Write-Host "=== [3/6] Uploading dataset ===" -ForegroundColor Cyan
    aws s3 cp $DataPath "s3://$BUCKET/data/raw/creditcard.csv" --region $Region
    Assert-LastCommand "s3 cp dataset"
    Write-Host "  OK - Uploaded to s3://$BUCKET/data/raw/creditcard.csv"

    Write-Host ""
    Write-Host "=== [4/6] Building pipeline DAG ===" -ForegroundColor Cyan
    python src/pipelines/build_pipeline.py
    Assert-LastCommand "build_pipeline.py"
    Write-Host "  OK - Pipeline upserted: fraud-detection-pipeline"

    Write-Host ""
    Write-Host "=== [5/6] Starting pipeline execution ===" -ForegroundColor Cyan
    python src/pipelines/run_pipeline.py
    Assert-LastCommand "run_pipeline.py"
    Write-Host ""
    Write-Host "  Pipeline is running asynchronously (~25-35 min)." -ForegroundColor Yellow
    Write-Host "  Monitor: https://console.aws.amazon.com/sagemaker/home?region=$Region#/pipelines"
    Write-Host ""
    Write-Host "  Once the pipeline succeeds and a model is Approved, deploy the endpoint with:"
    Write-Host "  .\scripts\deploy.ps1 -Region $Region -EmailForAlerts $EmailForAlerts -SkipCDK -SkipPipeline" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host ""
    Write-Host "=== [3-5/6] Skipping pipeline ===" -ForegroundColor Yellow
}

# --- 6. Approve + deploy endpoint
if (-not $SkipEndpoint) {
    Write-Host ""
    Write-Host "=== [6/6] Approving model + deploying endpoint ===" -ForegroundColor Cyan

    $pkgsJson = aws sagemaker list-model-packages --model-package-group-name fraud-detection-model-group --sort-by CreationTime --sort-order Descending --max-results 1 --region $Region --output json
    Assert-LastCommand "list-model-packages"

    $pkgList = ($pkgsJson | ConvertFrom-Json).ModelPackageSummaryList
    if ($pkgList.Count -eq 0) {
        Write-Host "ERROR: No model packages found in fraud-detection-model-group" -ForegroundColor Red
        exit 1
    }

    $pkg       = $pkgList[0]
    $pkgArn    = $pkg.ModelPackageArn
    $pkgStatus = $pkg.ModelApprovalStatus
    Write-Host "  Package : $pkgArn"
    Write-Host "  Status  : $pkgStatus"

    if ($pkgStatus -eq "PendingManualApproval") {
        Write-Host "  Auto-approving..."
        aws sagemaker update-model-package --model-package-arn $pkgArn --model-approval-status Approved --approval-description "Auto-approved by deploy.ps1" --region $Region | Out-Null
        Assert-LastCommand "update-model-package"
        Write-Host "  OK - Approved"
    } elseif ($pkgStatus -eq "Approved") {
        Write-Host "  OK - Already approved"
    } else {
        Write-Host "ERROR: Package status is '$pkgStatus' - cannot deploy" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "  Deploying endpoint: $EndpointName (~5-8 min)..."
    python src/deploy/deploy_endpoint.py --region $Region --role-arn $ROLE_ARN --artifact-bucket $BUCKET --endpoint-name $EndpointName --instance-type $InstanceType --canary-pct 100 --sns-topic-arn $SNS_ARN --wait
    Assert-LastCommand "deploy_endpoint.py"
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "  Bucket   : $BUCKET"
Write-Host "  Role ARN : $ROLE_ARN"
Write-Host "  Endpoint : $EndpointName"
Write-Host ""
Write-Host "  Test the endpoint:"
Write-Host "  '{\"instances\":[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,149.62]]}' | Out-File -Encoding ascii payload.json"
Write-Host "  aws sagemaker-runtime invoke-endpoint --endpoint-name $EndpointName --content-type application/json --body fileb://payload.json out.json --region $Region"
Write-Host "  Get-Content out.json"
Write-Host ""
Write-Host "  NOTE: Confirm SNS subscription at $EmailForAlerts" -ForegroundColor Yellow
