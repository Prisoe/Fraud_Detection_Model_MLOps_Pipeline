
---

## 

![img.png](images/architecture.png)

```md
# Architecture

## High-level flow

Developer
  |
  | 1) deploy.ps1
  v
CDK Stack (infra/)
  - S3 Artifacts Bucket
  - IAM SageMaker Execution Role
  - SNS Topic (email subscription)
  - EventBridge Rules
  - Lambda Alerts Formatter (pretty emails)
  |
  | 2) build_pipeline.py + run_pipeline.py
  v
SageMaker Pipeline
  - Preprocess step (Processing job)
  - Train step (Training job)
  - Evaluate step (Processing job -> evaluation.json)
  - RegisterModel step (Model Package Group)
  |
  v
Model Registry (mlops-blueprint-model-group)
  - Approved package version
  |
  | 3) deploy-endpoint.ps1
  v
SageMaker Endpoint (real-time, DataCapture enabled)
  |
  | 4) invoke-endpoint
  v
Predictions + Data Capture to S3

Monitoring
  - model_monitor_setup.py calculates PSI
  - publishes CloudWatch metric: MLOpsBlueprint/Drift OverallPSI_Max
  - CloudWatch Alarm -> SNS email