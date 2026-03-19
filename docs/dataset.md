# Dataset (sample_data.csv)

This repo uses `ml/sample_data.csv` as a small demo dataset to prove the pipeline works end-to-end.

## Format
- CSV with numeric feature columns and a label column.

Example columns:
- `f1`, `f2`, `f3` (features)
- `label` (target)

## How it’s used
1. `deploy.ps1` uploads the file to:
   - `s3://<ARTIFACT_BUCKET>/data/raw/sample_data.csv`
2. Pipeline preprocess step reads this file, performs split/transform, and writes:
   - train/val/test outputs to the pipeline execution S3 prefix.

## Splits
- Preprocess step splits into train/validation (and optionally test depending on your code).
- The exact split ratios are defined in your preprocess script (document them here after you confirm the values).