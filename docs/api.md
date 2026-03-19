
---

## docs/api.md (endpoint invocation + payloads)

```md
# API Instructions (SageMaker Endpoint)

## Invoke with AWS CLI

```powershell
$ENDPOINT="aws-mlops-blueprint-endpoint"
$REGION="us-east-1"

'{"instances":[[1.0,2.0,3.0]]}' | Out-File -Encoding ascii payload.json

aws sagemaker-runtime invoke-endpoint `
  --endpoint-name $ENDPOINT `
  --content-type "application/json" `
  --accept "application/json" `
  --body fileb://payload.json `
  out.json `
  --region $REGION

Get-Content out.json