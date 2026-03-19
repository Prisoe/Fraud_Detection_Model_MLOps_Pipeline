"""
Data Validation Step — runs before Preprocess
==============================================
Checks schema, statistics, and data quality of incoming creditcard.csv.
Fails fast with a clear error if anything is wrong — before wasting
compute on a training job that will produce a bad model.

Checks performed:
  1. Schema: required columns present, correct dtypes
  2. Row count: enough data to train (configurable min)
  3. Fraud rate: within expected range (0.05% – 5%)
  4. Missing values: no nulls in feature columns
  5. Range checks: V1-V28 within [-150, 150], Amount >= 0
     NOTE: EDA confirmed actual V ranges reach ±114 (V5), +121 (V7), ±54 (V20).
     The old [-30,30] assumption was wrong — these are PCA components and
     their scale depends on the original feature variances.
  6. Duplicate transactions: warn if > 1% duplicates
  7. Class balance: both classes present

Outputs:
  - /opt/ml/processing/validation/validation_report.json
    (read by ConditionStep to gate the pipeline)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def find_csv(path: str) -> str:
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return path
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.lower().endswith(".csv"))
        if not files:
            raise FileNotFoundError(f"No CSV in: {path}")
        return os.path.join(path, files[0])
    raise FileNotFoundError(f"Path not found: {path}")


def run_checks(df: pd.DataFrame, args) -> dict:
    checks = []
    passed = True

    def record(name: str, ok: bool, message: str, severity: str = "ERROR"):
        nonlocal passed
        status = "PASS" if ok else severity
        if not ok and severity == "ERROR":
            passed = False
        checks.append({"check": name, "status": status, "message": message})
        icon = "✅" if ok else ("⚠️" if severity == "WARN" else "❌")
        print(f"  {icon} [{status}] {name}: {message}")

    print("\n[validate] Running data quality checks...")

    # ── 1. Required columns
    required = {"Class", "Amount"} | {f"V{i}" for i in range(1, 29)}
    missing_cols = required - set(df.columns)
    record(
        "required_columns",
        len(missing_cols) == 0,
        f"All {len(required)} required columns present"
        if not missing_cols else f"Missing: {sorted(missing_cols)}",
    )

    # ── 2. Row count
    n = len(df)
    ok = n >= args.min_rows
    record("min_row_count", ok, f"{n:,} rows (min={args.min_rows:,})")

    # ── 3. Target column exists and is binary
    if "Class" in df.columns:
        classes = set(df["Class"].dropna().unique().tolist())
        ok = classes <= {0, 1}
        record("binary_target", ok, f"Class values: {sorted(classes)}")

        # ── 4. Fraud rate
        fraud_rate = float(df["Class"].mean())
        ok = args.min_fraud_rate <= fraud_rate <= args.max_fraud_rate
        record(
            "fraud_rate_range",
            ok,
            f"{fraud_rate:.4%} (expected {args.min_fraud_rate:.2%}–{args.max_fraud_rate:.2%})",
            severity="WARN" if not ok else "ERROR",
        )

        # ── 5. Both classes present
        n_fraud = int(df["Class"].sum())
        n_legit = int((df["Class"] == 0).sum())
        record("both_classes_present", n_fraud > 0 and n_legit > 0,
               f"Fraud={n_fraud:,}  Legitimate={n_legit:,}")

    # ── 6. Missing values in feature columns (raw CSV columns only)
    # IMPORTANT: only check columns that exist in the raw CSV.
    # Do NOT include any derived columns (time_hour, log_amount etc.)
    # EDA confirmed: the raw creditcard.csv dataset has zero nulls.
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if "Class" in df.columns:
        feature_cols.append("Class")
    feature_cols = [c for c in feature_cols if c in df.columns]
    null_counts = df[feature_cols].isnull().sum()
    total_nulls = int(null_counts.sum())
    ok = total_nulls == 0
    worst = null_counts.nlargest(3).to_dict() if total_nulls > 0 else {}
    record("no_feature_nulls", ok,
           f"0 nulls in features" if ok else f"{total_nulls:,} nulls. Worst: {worst}")

    # ── 7. V1-V28 range check
    # EDA-confirmed actual ranges (creditcard.csv):
    #   V5:  [-113.7, +34.8]   V6:  [-26.2, +73.3]   V7:  [-43.6, +120.6]
    #   V20: [-54.5, +39.4]    V1:  [-56.4, +2.5]     V8:  [-73.2, +20.0]
    # The old [-30,30] assumption was wrong. Using [-150, 150] as a
    # corruption-detection bound (10× the typical stddev of ~1.0).
    V_RANGE_MIN, V_RANGE_MAX = -150, 150
    v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
    if v_cols:
        v_min = float(df[v_cols].min().min())
        v_max = float(df[v_cols].max().max())
        ok = v_min >= V_RANGE_MIN and v_max <= V_RANGE_MAX
        record("feature_range_v_columns", ok,
               f"V1-V28 range: [{v_min:.2f}, {v_max:.2f}] (expected [{V_RANGE_MIN},{V_RANGE_MAX}])",
               severity="WARN")

    # ── 8. Amount >= 0
    if "Amount" in df.columns:
        neg_amount = int((df["Amount"] < 0).sum())
        record("amount_non_negative", neg_amount == 0,
               f"0 negative amounts" if neg_amount == 0 else f"{neg_amount:,} negative Amount values")

    # ── 9. Duplicate rows
    n_dupes = int(df.duplicated().sum())
    dupe_rate = n_dupes / max(n, 1)
    ok = dupe_rate <= 0.01
    record("duplicate_rate", ok,
           f"{n_dupes:,} duplicates ({dupe_rate:.2%}) — threshold 1%",
           severity="WARN")

    # ── 10. No constant features
    if v_cols:
        constant = [c for c in v_cols if df[c].nunique() <= 1]
        record("no_constant_features", len(constant) == 0,
               f"All V features have variance"
               if not constant else f"Constant columns: {constant}")

    return {
        "passed": passed,
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["status"] == "PASS"),
        "failed_checks": sum(1 for c in checks if c["status"] == "ERROR"),
        "warned_checks": sum(1 for c in checks if c["status"] == "WARN"),
        "checks": checks,
        "summary": {
            "n_rows": n,
            "n_fraud": int(df["Class"].sum()) if "Class" in df.columns else None,
            "fraud_rate": float(df["Class"].mean()) if "Class" in df.columns else None,
            "n_columns": len(df.columns),
        },
        # Key read by ConditionStep quality gate
        "validation_passed": int(passed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-data",      required=True)
    ap.add_argument("--output-dir",      required=True)
    ap.add_argument("--min-rows",        type=int,   default=10_000)
    ap.add_argument("--min-fraud-rate",  type=float, default=0.0005)
    ap.add_argument("--max-fraud-rate",  type=float, default=0.05)
    ap.add_argument("--fail-on-warn",    action="store_true",
                    help="Treat WARN as ERROR (strict mode)")
    args = ap.parse_args()

    csv_path = find_csv(args.input_data)
    print(f"[validate] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[validate] Shape: {df.shape}")

    report = run_checks(df, args)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "validation_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n[validate] Report → {out_path}")
    print(f"[validate] Result: {'✅ PASSED' if report['passed'] else '❌ FAILED'} "
          f"({report['passed_checks']}/{report['total_checks']} checks passed, "
          f"{report['warned_checks']} warnings)")

    if not report["passed"]:
        sys.exit(1)

    print("✅ Validation complete")


if __name__ == "__main__":
    main()
