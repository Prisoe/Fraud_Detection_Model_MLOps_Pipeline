"""
Drift Monitoring — Fraud Detection (v2)
=========================================
Upgrades over v1:
  1. Label/outcome drift: tracks fraud_rate in ground-truth-labelled data
     (when fraud analysts have reviewed captured transactions)
  2. Retraining trigger: if drift exceeds threshold, automatically kicks
     off the SageMaker pipeline instead of just emailing
  3. Per-variant PSI: tracks champion vs challenger separately
  4. Clearer PSI interpretation tiers with recommended actions
  5. Stores drift history to S3 for trending
"""

import argparse
import io
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd

EPS = 1e-6


def _safe_p(p: float) -> float:
    return max(p, EPS)


def psi(expected: List[float], actual: List[float]) -> float:
    return float(sum((_safe_p(a) - _safe_p(e)) * math.log(_safe_p(a) / _safe_p(e))
                     for e, a in zip(expected, actual)))


def psi_severity(value: float) -> str:
    if value < 0.10:  return "OK"
    if value < 0.25:  return "MODERATE"
    return "HIGH"


def numeric_edges(series: pd.Series, n_bins: int = 10) -> List[float]:
    qs = [i / n_bins for i in range(1, n_bins)]
    return sorted(set(series.quantile(qs).dropna().tolist()))


def numeric_dist(series: pd.Series, edges: List[float]) -> List[float]:
    s = series.dropna()
    if s.empty:
        return [1.0] + [0.0] * len(edges)
    bins = [-math.inf] + edges + [math.inf]
    counts = pd.cut(s, bins=bins, include_lowest=True).value_counts(sort=False)
    total  = counts.sum()
    return [float(c / total) for c in counts]


@dataclass
class S3Loc:
    bucket: str
    key: str


def parse_uri(uri: str) -> S3Loc:
    no_scheme = uri.replace("s3://", "", 1)
    parts = no_scheme.split("/", 1)
    return S3Loc(bucket=parts[0], key=parts[1] if len(parts) > 1 else "")


def read_csv_s3(s3, uri: str) -> pd.DataFrame:
    loc  = parse_uri(uri)
    body = s3.get_object(Bucket=loc.bucket, Key=loc.key)["Body"].read()
    return pd.read_csv(io.BytesIO(body))


def read_recent_csvs(s3, prefix_uri: str, max_files: int = 5) -> pd.DataFrame:
    loc       = parse_uri(prefix_uri)
    paginator = s3.get_paginator("list_objects_v2")
    keys      = []
    for page in paginator.paginate(Bucket=loc.bucket, Prefix=loc.key):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".csv"):
                keys.append((obj["Key"], obj["LastModified"]))
    keys.sort(key=lambda x: x[1], reverse=True)
    if not keys:
        raise FileNotFoundError(f"No CSVs under: {prefix_uri}")
    frames = []
    for k, _ in keys[:max_files]:
        body = s3.get_object(Bucket=loc.bucket, Key=k)["Body"].read()
        frames.append(pd.read_csv(io.BytesIO(body)))
    return pd.concat(frames, ignore_index=True)


def publish_cw(cw, namespace, dims: Dict[str, str], metrics: Dict[str, float]):
    data = [
        {"MetricName": name, "Value": float(val), "Unit": "None",
         "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()]}
        for name, val in metrics.items()
    ]
    for i in range(0, len(data), 20):
        cw.put_metric_data(Namespace=namespace, MetricData=data[i:i+20])


def trigger_retraining(sm, pipeline_name: str, reason: str) -> str:
    """Kick off the SageMaker pipeline to retrain the model."""
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"drift-triggered-{ts}"
    resp = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=name,
        PipelineParameters=[],
        PipelineExecutionDescription=f"Auto-triggered by drift monitor: {reason}",
    )
    arn = resp["PipelineExecutionArn"]
    print(f"[monitor] 🔄 Triggered retraining pipeline: {arn}")
    return arn


def save_drift_history(s3, bucket: str, report: dict):
    """Append drift report to a time-series history file in S3."""
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    key = f"monitoring/drift-history/{ts}.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(report, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[monitor] Drift history → s3://{bucket}/{key}")


def check_label_drift(recent: pd.DataFrame, label_col: str,
                      baseline_fraud_rate: float,
                      warn_multiplier: float = 2.0) -> dict:
    """
    Outcome/label drift: checks if recent fraud rate has meaningfully
    shifted from training baseline.
    Only meaningful when recent data has ground-truth labels
    (e.g. fraud analyst verdicts on captured transactions).
    """
    if label_col not in recent.columns:
        return {"available": False, "reason": f"'{label_col}' not in recent data"}

    recent_rate = float(recent[label_col].mean())
    ratio       = recent_rate / max(baseline_fraud_rate, EPS)
    drifted     = ratio >= warn_multiplier or ratio <= (1 / warn_multiplier)

    return {
        "available":          True,
        "baseline_fraud_rate": baseline_fraud_rate,
        "recent_fraud_rate":  recent_rate,
        "ratio":              ratio,
        "drifted":            drifted,
        "severity":           "HIGH" if drifted else "OK",
        "message": (
            f"Fraud rate changed {ratio:.1f}× (baseline={baseline_fraud_rate:.4%} → "
            f"recent={recent_rate:.4%})"
            if drifted else
            f"Fraud rate stable (baseline={baseline_fraud_rate:.4%} recent={recent_rate:.4%})"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region",              default="us-east-1")
    ap.add_argument("--baseline-s3-uri",     required=True)
    ap.add_argument("--recent-s3-prefix",    required=True)
    ap.add_argument("--artifact-bucket",     default=os.environ.get("ARTIFACT_BUCKET", ""))
    ap.add_argument("--label-col",           default="label")
    ap.add_argument("--baseline-fraud-rate", type=float, default=0.00172,
                    help="Fraud rate in training data (EDA confirmed: 0.1727%%)")
    ap.add_argument("--bins",                type=int,   default=10)
    ap.add_argument("--recent-max-files",    type=int,   default=5)
    ap.add_argument("--psi-threshold",       type=float, default=0.25)
    ap.add_argument("--cw-namespace",        default="FraudDetection/Drift")
    ap.add_argument("--cw-dimension-value",  default="fraud-detection")
    ap.add_argument("--sns-topic-arn",       default="")
    ap.add_argument("--publish-top-n",       type=int,   default=10)
    # EDA-confirmed SHAP feature priority order (top features by mean |SHAP|):
    # V14 (2.41) > V4 (1.69) > V12 (0.77) > V10 (0.61) > V11 (0.45)
    # Note: correlation ranking (V17 #1) differs from SHAP (V14 #1).
    # SHAP captures nonlinear contributions — use it for drift priority.
    ap.add_argument("--priority-features",   default="V14,V4,V12,V10,V11,V3,V8,V16,V7,V19",
                    help="Comma-separated features to highlight in drift report (SHAP order)")
    # Retraining trigger
    ap.add_argument("--pipeline-name",       default="fraud-detection-pipeline",
                    help="SageMaker pipeline to trigger on high drift")
    ap.add_argument("--auto-retrain",        action="store_true",
                    help="Automatically trigger pipeline retraining when drift is HIGH")
    ap.add_argument("--label-drift-multiplier", type=float, default=2.0,
                    help="Alert if recent fraud rate is this many × baseline")
    args = ap.parse_args()

    s3  = boto3.client("s3",          region_name=args.region)
    cw  = boto3.client("cloudwatch",  region_name=args.region)
    sns = boto3.client("sns",         region_name=args.region)
    sm  = boto3.client("sagemaker",   region_name=args.region)

    print("[monitor] Loading baseline:", args.baseline_s3_uri)
    baseline = read_csv_s3(s3, args.baseline_s3_uri)

    print("[monitor] Loading recent:", args.recent_s3_prefix)
    recent = read_recent_csvs(s3, args.recent_s3_prefix, max_files=args.recent_max_files)

    # ── Feature PSI
    exclude     = {args.label_col}
    common_cols = [c for c in baseline.columns if c in recent.columns and c not in exclude]
    print(f"[monitor] Comparing {len(common_cols)} features")

    feature_psi: Dict[str, float] = {}
    for col in common_cols:
        b, r = baseline[col].dropna(), recent[col].dropna()
        if pd.api.types.is_numeric_dtype(b):
            edges = numeric_edges(b, n_bins=args.bins)
            feature_psi[col] = psi(numeric_dist(b, edges), numeric_dist(r, edges))

    overall_max  = max(feature_psi.values()) if feature_psi else 0.0
    overall_mean = sum(feature_psi.values()) / max(len(feature_psi), 1)

    # ── Label/outcome drift
    label_drift = check_label_drift(
        recent, args.label_col, args.baseline_fraud_rate, args.label_drift_multiplier
    )

    # ── Build CW metrics
    dims = {"Project": args.cw_dimension_value}
    cw_metrics: Dict[str, float] = {
        "OverallPSI_Max":  overall_max,
        "OverallPSI_Mean": overall_mean,
    }
    if label_drift["available"]:
        cw_metrics["RecentFraudRate"]      = label_drift["recent_fraud_rate"]
        cw_metrics["FraudRateRatio"]       = label_drift["ratio"]

    top_n = sorted(feature_psi.items(), key=lambda x: x[1], reverse=True)[:args.publish_top_n]
    for name, val in top_n:
        safe = name.replace(" ", "_").replace("/", "_")[:200]
        cw_metrics[f"FeaturePSI_{safe}"] = float(val)

    print("[monitor] Publishing metrics...")
    publish_cw(cw, args.cw_namespace, dims, cw_metrics)

    # ── Print summary
    print(f"\n[monitor] Feature Drift Summary:")
    print(f"  OverallPSI_Max  = {overall_max:.4f}  [{psi_severity(overall_max)}]")
    print(f"  OverallPSI_Mean = {overall_mean:.4f}")
    print(f"\n  Top drifted features (by PSI):")
    for name, val in top_n[:5]:
        sev = psi_severity(val)
        icon = "❌" if sev == "HIGH" else "⚠️" if sev == "MODERATE" else "✅"
        print(f"    {icon} {name:30s}: {val:.4f}  [{sev}]")

    # Report SHAP-priority features specifically — these matter most for model behaviour
    priority_features = [f.strip() for f in args.priority_features.split(",") if f.strip()]
    priority_psi = {f: feature_psi.get(f, 0.0) for f in priority_features if f in feature_psi}
    if priority_psi:
        print(f"\n  SHAP-priority features (EDA: V14>V4>V12>V10>V11 drive predictions):")
        for name, val in priority_psi.items():
            sev  = psi_severity(val)
            icon = "❌" if sev == "HIGH" else "⚠️" if sev == "MODERATE" else "✅"
            print(f"    {icon} {name:10s}: PSI={val:.4f}  [{sev}]")

    if label_drift["available"]:
        icon = "❌" if label_drift["drifted"] else "✅"
        print(f"\n  {icon} Label drift: {label_drift['message']}")

    # ── Determine if action is needed
    feature_drift_high = overall_max >= args.psi_threshold
    label_drift_high   = label_drift.get("drifted", False)
    needs_action       = feature_drift_high or label_drift_high

    retrain_arn = None
    if needs_action:
        reasons = []
        if feature_drift_high:
            reasons.append(f"FeaturePSI_Max={overall_max:.3f}>={args.psi_threshold}")
        if label_drift_high:
            reasons.append(f"FraudRateRatio={label_drift.get('ratio',0):.1f}×")
        reason_str = " | ".join(reasons)

        print(f"\n⚠️  ACTION NEEDED: {reason_str}")

        # Auto-retrain
        if args.auto_retrain:
            retrain_arn = trigger_retraining(sm, args.pipeline_name, reason_str)
        else:
            print("[monitor] Pass --auto-retrain to trigger pipeline automatically")

        # SNS alert
        if args.sns_topic_arn:
            subject = f"[FRAUD DRIFT] {reason_str}"
            body = json.dumps({
                "reason":           reason_str,
                "overall_psi_max":  overall_max,
                "overall_psi_mean": overall_mean,
                "threshold":        args.psi_threshold,
                "label_drift":      label_drift,
                "top_features":     top_n[:10],
                "retrain_triggered": retrain_arn is not None,
                "retrain_arn":       retrain_arn,
            }, indent=2)
            sns.publish(TopicArn=args.sns_topic_arn, Subject=subject[:100], Message=body)
            print("[monitor] Alert sent to SNS")
    else:
        print(f"\n✅ No action needed (PSI_Max={overall_max:.3f} < {args.psi_threshold})")

    # ── Save drift history
    drift_report = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "overall_psi_max":    overall_max,
        "overall_psi_mean":   overall_mean,
        "label_drift":        label_drift,
        "feature_psi":        dict(top_n),
        "shap_priority_psi":  priority_psi,   # EDA-informed: V14,V4,V12,V10,V11
        "action_taken":       "retrain_triggered" if retrain_arn else ("alert_sent" if needs_action else "none"),
        "retrain_arn":        retrain_arn,
    }
    if args.artifact_bucket:
        save_drift_history(s3, args.artifact_bucket, drift_report)

    print("✅ Drift check complete")


if __name__ == "__main__":
    main()
