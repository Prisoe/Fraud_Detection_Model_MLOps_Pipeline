# Model Evaluation

The pipeline writes evaluation results to S3 as `evaluation.json`.

## Metrics
Document the metrics your `Evaluate` step computes (example):
- Accuracy
- Precision / Recall
- F1-score
- Confusion matrix
- Any regression metrics if relevant

## Where results live in S3
You can find evaluation output via the model package metadata or pipeline execution artifacts.

Example from your model package:
`s3://<ARTIFACT_BUCKET>/<pipeline-name>/<execution-id>/Evaluate/output/evaluation/evaluation.json`,
's3://mlopsblueprintstack-mlopsartifactsbucketed627dbf-lniwlwdmqzm3/mlops-blueprint-pipeline/301zhrf6eh4y/Evaluate/output/evaluation/evaluation.json'

## How to view it
```bash
aws s3 cp s3://<...>/evaluation.json - | cat