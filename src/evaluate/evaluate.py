"""
Evaluation — Fraud Detection (v2)
"""
import subprocess, sys
# Install xgboost before importing it — sklearn container doesn't include it
subprocess.run([sys.executable, "-m", "pip", "install", "xgboost>=1.7.0", "-q"],
               check=True)

import argparse
import json
import os
import tarfile

import boto3
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# How much better does the challenger need to be to replace the champion?
CHAMPION_IMPROVEMENT_MARGIN = 0.02   # +2 AUPRC points minimum


def extract_model(model_dir: str) -> str:
    tar = os.path.join(model_dir, "model.tar.gz")
    if not os.path.exists(tar):
        return model_dir
    out = os.path.join(model_dir, "extracted")
    os.makedirs(out, exist_ok=True)
    with tarfile.open(tar, "r:gz") as t:
        t.extractall(path=out)
    return out


def find_file(root: str, filename: str) -> str:
    for candidate in [
        os.path.join(root, filename),
        os.path.join(root, "model", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{filename} not found under {root}. Contents: {os.listdir(root)}")


def load_threshold(model_root: str) -> float:
    """Load the optimal threshold saved during training. Falls back to 0.5."""
    try:
        path = find_file(model_root, "threshold.json")
        with open(path) as fh:
            data = json.load(fh)
        t = float(data["threshold"])
        print(f"[evaluate] Loaded threshold={t:.4f} from threshold.json "
              f"(method={data.get('selection_method','?')}, "
              f"FN_cost={data.get('fn_cost','?')}× FP_cost={data.get('fp_cost','?')}×)")
        return t
    except FileNotFoundError:
        print("[evaluate] threshold.json not found — using 0.5 fallback")
        return 0.5


def load_shap_summary(model_root: str) -> dict:
    try:
        path = find_file(model_root, "shap_summary.json")
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def get_champion_auprc(region: str, model_package_group: str) -> Optional[float]:
    """
    Fetch the AUPRC of the currently Approved model in the registry.
    Returns None if no champion exists yet (first run).
    """
    try:
        sm = boto3.client("sagemaker", region_name=region)
        resp = sm.list_model_packages(
            ModelPackageGroupName=model_package_group,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=50,
        )
        for pkg in resp.get("ModelPackageSummaryList", []):
            if (pkg.get("ModelPackageStatus") == "Completed"
                    and pkg.get("ModelApprovalStatus") == "Approved"):
                arn  = pkg["ModelPackageArn"]
                desc = sm.describe_model_package(ModelPackageName=arn)
                s3_uri = (
                    desc.get("ModelMetrics", {})
                        .get("ModelStatistics", {})
                        .get("S3Uri", "")
                )
                if s3_uri:
                    import io
                    s3 = boto3.client("s3", region_name=region)
                    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
                    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    metrics = json.loads(body)
                    champ_auprc = metrics.get("avg_precision")
                    if champ_auprc is not None:
                        print(f"[evaluate] Champion model: {arn}")
                        print(f"[evaluate] Champion AUPRC: {champ_auprc:.4f}")
                        return float(champ_auprc)
    except Exception as e:
        print(f"[evaluate] Could not fetch champion metrics: {e} — proceeding without comparison")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",                  required=True)
    ap.add_argument("--test",                   required=True)
    ap.add_argument("--output-dir",             required=True)
    ap.add_argument("--region",                 default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--model-package-group",    default="fraud-detection-model-group")
    ap.add_argument("--fn-cost",                type=float, default=10.0)
    ap.add_argument("--fp-cost",                type=float, default=1.0)
    ap.add_argument("--skip-champion-check",    action="store_true",
                    help="Skip champion/challenger comparison (use for first run)")
    args = ap.parse_args()

    # ── Load model bundle
    model_root = extract_model(args.model)
    model_path = find_file(model_root, "model.joblib")
    model      = joblib.load(model_path)
    threshold  = load_threshold(model_root)
    shap_info  = load_shap_summary(model_root)
    print(f"[evaluate] Loaded model from: {model_path}")

    # ── Load test data
    test_csv = os.path.join(args.test, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Missing test.csv at: {test_csv}")
    test_df   = pd.read_csv(test_csv)
    X_test    = test_df.drop(columns=["label"]).values
    y_test    = test_df["label"].values
    n_fraud   = int(y_test.sum())
    print(f"[evaluate] Test: {len(y_test):,} rows | Fraud: {n_fraud:,} ({n_fraud/len(y_test):.4%})")

    # ── Predict
    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)

    # ── Core metrics
    avg_precision = float(average_precision_score(y_test, y_prob))
    roc_auc       = float(roc_auc_score(y_test, y_prob))
    f1            = float(f1_score(y_test, y_pred, zero_division=0))
    precision     = float(precision_score(y_test, y_pred, zero_division=0))
    recall        = float(recall_score(y_test, y_pred, zero_division=0))
    cm            = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    print(f"\n[evaluate] Test metrics (threshold={threshold:.4f}):")
    print(f"  avg_precision (AUPRC) : {avg_precision:.4f}  ← quality gate")
    print(f"  roc_auc               : {roc_auc:.4f}")
    print(f"  f1                    : {f1:.4f}")
    print(f"  precision             : {precision:.4f}  ({tp} TP / {tp+fp} flagged)")
    print(f"  recall                : {recall:.4f}  ({tp} TP / {tp+fn} actual fraud)")
    print(f"  confusion matrix      : TN={tn} FP={fp} FN={fn} TP={tp}")

    # ── Business cost estimate
    business_cost = args.fn_cost * fn + args.fp_cost * fp
    print(f"\n[evaluate] Business cost estimate:")
    print(f"  FN={fn} × {args.fn_cost} + FP={fp} × {args.fp_cost} = {business_cost:.0f} units")

    # ── Champion/challenger comparison
    champion_auprc  = None
    beats_champion  = True
    champion_delta  = None

    if not args.skip_champion_check:
        champion_auprc = get_champion_auprc(args.region, args.model_package_group)
        if champion_auprc is not None:
            champion_delta = avg_precision - champion_auprc
            beats_champion = champion_delta >= CHAMPION_IMPROVEMENT_MARGIN
            print(f"\n[evaluate] Champion/Challenger:")
            print(f"  Champion AUPRC  : {champion_auprc:.4f}")
            print(f"  Challenger AUPRC: {avg_precision:.4f}")
            print(f"  Delta           : {champion_delta:+.4f} (min required: +{CHAMPION_IMPROVEMENT_MARGIN})")
            if beats_champion:
                print(f"  ✅ Challenger beats champion — will proceed to registration")
            else:
                print(f"  ❌ Challenger does NOT meet improvement bar — will block registration")
        else:
            print("\n[evaluate] No champion found — first run, skipping comparison")

    # ── Write evaluation.json
    report = classification_report(y_test, y_pred, output_dict=True)
    output = {
        # Keys read by pipeline ConditionSteps
        "avg_precision":    avg_precision,
        "beats_champion":   int(beats_champion),   # 1 = ok to register
        # Supporting metrics
        "roc_auc":          roc_auc,
        "f1":               f1,
        "precision":        precision,
        "recall":           recall,
        "threshold_used":   threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "business_cost": {
            "total":    float(business_cost),
            "fn_count": int(fn),
            "fp_count": int(fp),
            "fn_cost_weight": args.fn_cost,
            "fp_cost_weight": args.fp_cost,
        },
        "champion_comparison": {
            "champion_auprc":  champion_auprc,
            "challenger_auprc": avg_precision,
            "delta":           champion_delta,
            "improvement_margin_required": CHAMPION_IMPROVEMENT_MARGIN,
            "beats_champion":  beats_champion,
        },
        "shap": shap_info,
        "classification_report": report,
        "n_test":       int(len(y_test)),
        "n_fraud_test": n_fraud,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "evaluation.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n[evaluate] Wrote evaluation.json → {out_path}")
    print("✅ Evaluation complete")


if __name__ == "__main__":
    main()
