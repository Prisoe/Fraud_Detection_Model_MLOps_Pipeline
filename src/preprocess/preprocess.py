"""
Step 2 — Preprocessing for Credit Card Fraud Detection
========================================================
Input  : creditcard.csv  (284k rows, V1-V28, Time, Amount, Class)
Output :
  - /opt/ml/processing/train/train.csv
  - /opt/ml/processing/val/val.csv
  - /opt/ml/processing/test/test.csv
  - /opt/ml/processing/baseline/baseline.csv   ← drift monitoring reference

Key decisions vs generic template:
  - Drop 'Time' (elapsed-seconds artifact, not meaningful at inference time)
  - Scale 'Amount' with RobustScaler (heavy right tail, outliers)
  - Stratified splits to preserve 0.17% fraud rate in every split
  - Write a baseline CSV (train split, features only) for PSI drift monitoring
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def find_csv(path: str) -> str:
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return path
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"No CSV in: {path}")
        return os.path.join(path, sorted(files)[0])
    raise FileNotFoundError(f"Path not found: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-data",    required=True)
    ap.add_argument("--output-train",  required=True)
    ap.add_argument("--output-val",    required=True)
    ap.add_argument("--output-test",   required=True)
    ap.add_argument("--output-baseline", required=False, default=None,
                    help="Optional path for drift-monitoring baseline CSV")
    ap.add_argument("--test-size",     type=float, default=0.15)
    ap.add_argument("--val-size",      type=float, default=0.15)
    ap.add_argument("--random-state",  type=int,   default=42)
    args = ap.parse_args()

    csv_path = find_csv(args.input_data)
    print(f"[preprocess] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[preprocess] Raw shape: {df.shape}")

    # ---- Validate schema
    required_cols = {"Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

    # ---- Drop duplicates
    # EDA confirmed 1,081 duplicates (0.38%) in creditcard.csv.
    # Dropping before splitting prevents the same row appearing in both
    # train and test, which would cause data leakage and inflated metrics.
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[preprocess] Dropped {n_dropped:,} duplicate rows ({n_dropped/n_before:.3%})")

    # ---- Drop 'Time'
    # EDA confirmed: |corr(Time, Class)| = 0.012 — negligible linear signal.
    # Fraud rate by hour chart shows some hourly variation but the overall
    # correlation is too weak to justify the inference-time complexity.
    # Dropping keeps the feature count at 29 (V1-V28 + Amount).
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
        print("[preprocess] Dropped 'Time' (EDA: |corr|=0.012 with Class)")

    # ---- Rename 'Class' → 'label' (pipeline convention)
    df = df.rename(columns={"Class": "label"})

    # ---- Scale 'Amount' with RobustScaler (resistant to fraud-amount outliers)
    scaler = RobustScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    print("[preprocess] Scaled 'Amount' with RobustScaler")

    # ---- Class distribution
    class_counts = df["label"].value_counts()
    print(f"[preprocess] Class distribution:\n{class_counts.to_string()}")
    fraud_rate = class_counts.get(1, 0) / len(df)
    print(f"[preprocess] Fraud rate: {fraud_rate:.4%}")

    # ---- Stratified split: train / temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(args.test_size + args.val_size),
        random_state=args.random_state,
        stratify=df["label"],
    )

    # ---- Stratified split: val / test from temp
    val_ratio = args.val_size / (args.test_size + args.val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        random_state=args.random_state,
        stratify=temp_df["label"],
    )

    print(f"[preprocess] Split → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = split["label"].sum() / len(split)
        print(f"  {name} fraud rate: {rate:.4%}")

    # ---- Write splits
    for out_dir, split, fname in [
        (args.output_train, train_df, "train.csv"),
        (args.output_val,   val_df,   "val.csv"),
        (args.output_test,  test_df,  "test.csv"),
    ]:
        os.makedirs(out_dir, exist_ok=True)
        split.to_csv(os.path.join(out_dir, fname), index=False)

    # ---- Baseline CSV for drift monitoring (features only, no label)
    baseline_dir = args.output_baseline
    if baseline_dir:
        os.makedirs(baseline_dir, exist_ok=True)
        feature_cols = [c for c in train_df.columns if c != "label"]
        train_df[feature_cols].to_csv(os.path.join(baseline_dir, "baseline.csv"), index=False)
        print(f"[preprocess] Wrote baseline CSV to: {baseline_dir}")

    print("✅ Preprocessing complete")


if __name__ == "__main__":
    main()
