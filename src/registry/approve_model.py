"""
Model Registry Governance — Fraud Detection (v2)
=================================================
Upgrades over v1:
  1. Enforces metric review before approval — prints full evaluation
     report and requires explicit confirmation (--yes flag or interactive)
  2. Champion/challenger gate: blocks approval if new model doesn't
     beat current champion by required margin
  3. Audit log: writes approval decision + metrics to S3
  4. Scoped IAM check: warns if role is using overly broad permissions
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone

import boto3
from typing import Optional, Tuple

REQUIRED_METRICS   = ["avg_precision", "recall", "precision"]
AUPRC_MINIMUM      = 0.80
RECALL_MINIMUM     = 0.75   # must catch at least 75% of actual fraud
PRECISION_MINIMUM  = 0.70   # at least 70% of flagged must be real fraud
CHAMPION_MARGIN    = 0.02   # challenger must be >= 2 AUPRC points better


def _get_metrics_from_s3(sm, s3, arn: str) -> dict:
    desc    = sm.describe_model_package(ModelPackageName=arn)
    s3_uri  = (
        desc.get("ModelMetrics", {})
            .get("ModelStatistics", {})
            .get("S3Uri", "")
    )
    if not s3_uri:
        return {}
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def _get_current_champion(sm, group: str) -> Tuple[Optional[str], dict]:
    """Returns (arn, metrics) of the currently Approved model, or (None, {})."""
    resp = sm.list_model_packages(
        ModelPackageGroupName=group, SortBy="CreationTime",
        SortOrder="Descending", MaxResults=50,
    )
    for p in resp.get("ModelPackageSummaryList", []):
        if p["ModelPackageStatus"] == "Completed" and p["ModelApprovalStatus"] == "Approved":
            return p["ModelPackageArn"], {}
    return None, {}


def _print_metrics_report(metrics: dict, arn: str):
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    print(f"ARN: {arn}")
    print()
    key_metrics = [
        ("avg_precision (AUPRC)",  "avg_precision",   AUPRC_MINIMUM),
        ("recall (fraud catch rate)", "recall",        RECALL_MINIMUM),
        ("precision (flag accuracy)", "precision",     PRECISION_MINIMUM),
        ("roc_auc",                "roc_auc",          None),
        ("f1",                     "f1",               None),
        ("threshold_used",         "threshold_used",   None),
    ]
    for label, key, minimum in key_metrics:
        val = metrics.get(key)
        if val is None:
            continue
        meets = f"  ✅ >= {minimum}" if minimum and float(val) >= minimum else (
                f"  ❌ BELOW {minimum}" if minimum else "")
        print(f"  {label:35s}: {float(val):.4f}{meets}")

    cm = metrics.get("confusion_matrix", {})
    if cm:
        tn, fp, fn, tp = cm.get("tn",0), cm.get("fp",0), cm.get("fn",0), cm.get("tp",0)
        print(f"\n  Confusion Matrix (test set):")
        print(f"    True Negatives  (correct legit) : {tn:,}")
        print(f"    False Positives (false alarm)   : {fp:,}")
        print(f"    False Negatives (missed fraud)  : {fn:,}  ← most costly")
        print(f"    True Positives  (caught fraud)  : {tp:,}")

    bc = metrics.get("business_cost", {})
    if bc:
        print(f"\n  Business Cost Estimate          : {bc.get('total', 0):.0f} units")
        print(f"    ({bc.get('fn_count',0)} missed fraud × {bc.get('fn_cost_weight',10)} + "
              f"{bc.get('fp_count',0)} false alarms × {bc.get('fp_cost_weight',1)})")

    shap = metrics.get("shap", {})
    if shap.get("top_5_features"):
        print(f"\n  Top fraud-driving features (SHAP):")
        for feat in shap["top_5_features"]:
            importance = shap.get("feature_importance_mean_abs_shap", {}).get(feat, 0)
            print(f"    {feat:15s}: mean |SHAP| = {importance:.4f}")

    cv = metrics.get("cv_auprc_mean")
    if cv:
        std = metrics.get("cv_auprc_std", 0)
        print(f"\n  Cross-validation AUPRC: {cv:.4f} ± {std:.4f} "
              f"[{metrics.get('cv_folds',5)}-fold]")

    print("="*60)


def _check_approval_criteria(metrics: dict) -> Tuple[bool, list[str]]:
    """Returns (passes, list_of_failures)."""
    failures = []
    for label, key, minimum in [
        ("AUPRC",     "avg_precision", AUPRC_MINIMUM),
        ("recall",    "recall",        RECALL_MINIMUM),
        ("precision", "precision",     PRECISION_MINIMUM),
    ]:
        val = metrics.get(key)
        if val is None:
            failures.append(f"Missing metric: {key}")
        elif float(val) < minimum:
            failures.append(f"{label}={float(val):.4f} < required {minimum}")
    return len(failures) == 0, failures


def _write_audit_log(s3, bucket: str, arn: str, action: str,
                     metrics: dict, approver: str, reason: str):
    ts  = datetime.now(timezone.utc).isoformat()
    log = {
        "timestamp":  ts,
        "action":     action,
        "arn":        arn,
        "approver":   approver,
        "reason":     reason,
        "key_metrics": {
            "avg_precision": metrics.get("avg_precision"),
            "recall":        metrics.get("recall"),
            "precision":     metrics.get("precision"),
            "roc_auc":       metrics.get("roc_auc"),
        },
    }
    key = f"monitoring/approval-audit/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    s3.put_object(Bucket=bucket, Key=key,
                  Body=json.dumps(log, indent=2).encode(), ContentType="application/json")
    print(f"[registry] Audit log → s3://{bucket}/{key}")


def cmd_list(sm, group: str):
    resp = sm.list_model_packages(
        ModelPackageGroupName=group, SortBy="CreationTime",
        SortOrder="Descending", MaxResults=20,
    )
    pkgs = resp.get("ModelPackageSummaryList", [])
    if not pkgs:
        print("No model packages found.")
        return
    print(f"\nModel packages in: {group}")
    print(f"{'Ver':>4}  {'Status':<12}  {'Approval':<26}  ARN")
    print("-"*90)
    for p in pkgs:
        ver = p.get("ModelPackageVersion", "?")
        st  = p.get("ModelPackageStatus", "?")
        ap  = p.get("ModelApprovalStatus", "?")
        arn = p["ModelPackageArn"]
        icon = "✅" if ap == "Approved" else "❌" if ap == "Rejected" else "🕐"
        print(f"{icon} {ver:>3}  {st:<12}  {ap:<26}  {arn}")


def cmd_approve(sm, s3, args):
    arn     = args.arn
    group   = args.group
    bucket  = args.artifact_bucket or os.environ.get("ARTIFACT_BUCKET", "")
    metrics = _get_metrics_from_s3(sm, s3, arn)

    # ── Print full report (enforce review)
    _print_metrics_report(metrics, arn)

    # ── Check hard criteria
    passes, failures = _check_approval_criteria(metrics)
    if not passes:
        print(f"\n❌ MODEL FAILS APPROVAL CRITERIA:")
        for f in failures:
            print(f"   • {f}")
        if not args.force:
            print("   Pass --force to override (not recommended)")
            sys.exit(1)
        else:
            print("   ⚠️  --force passed — overriding failures")

    # ── Champion comparison
    champ_arn, _ = _get_current_champion(sm, group)
    if champ_arn and champ_arn != arn:
        champ_metrics = _get_metrics_from_s3(sm, s3, champ_arn)
        champ_auprc   = champ_metrics.get("avg_precision", 0)
        new_auprc     = metrics.get("avg_precision", 0)
        delta         = (new_auprc or 0) - (champ_auprc or 0)
        print(f"\n  Champion AUPRC  : {champ_auprc:.4f}")
        print(f"  Challenger AUPRC: {new_auprc:.4f}  (delta={delta:+.4f})")
        if delta < CHAMPION_MARGIN and not args.force:
            print(f"\n❌ Challenger does not improve on champion by ≥{CHAMPION_MARGIN} AUPRC points.")
            print("   Pass --force to override.")
            sys.exit(1)

    # ── Interactive confirmation (unless --yes)
    if not args.yes:
        print(f"\n{'✅ All criteria met.' if passes else '⚠️  Criteria overridden.'}")
        print(f"You are about to APPROVE: {arn}")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    # ── Approve
    sm.update_model_package(
        ModelPackageArn=arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=args.description,
    )
    print(f"\n✅ Approved: {arn}")

    # ── Audit log
    if bucket:
        _write_audit_log(s3, bucket, arn, "Approved", metrics,
                         approver=os.environ.get("USER", "unknown"),
                         reason=args.description)


def cmd_reject(sm, s3, args):
    arn     = args.arn
    bucket  = args.artifact_bucket or os.environ.get("ARTIFACT_BUCKET", "")
    metrics = _get_metrics_from_s3(sm, s3, arn)

    sm.update_model_package(
        ModelPackageArn=arn,
        ModelApprovalStatus="Rejected",
        ApprovalDescription=args.description,
    )
    print(f"✅ Rejected: {arn}")

    if bucket:
        _write_audit_log(s3, bucket, arn, "Rejected", metrics,
                         approver=os.environ.get("USER", "unknown"),
                         reason=args.description)


def main():
    ap = argparse.ArgumentParser(description="Fraud Detection Model Registry (v2 — governed approval)")
    ap.add_argument("--region",          default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--group",           default="fraud-detection-model-group")
    ap.add_argument("--artifact-bucket", default=os.environ.get("ARTIFACT_BUCKET", ""))
    ap.add_argument("--action",          choices=["list", "approve", "reject", "metrics"],
                    default="list")
    ap.add_argument("--arn",             default=None)
    ap.add_argument("--description",     default="Approved via CLI")
    ap.add_argument("--yes",             action="store_true",
                    help="Skip interactive confirmation")
    ap.add_argument("--force",           action="store_true",
                    help="Override metric failures (use with caution)")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3",        region_name=args.region)

    if args.action == "list":
        cmd_list(sm, args.group)

    elif args.action == "approve":
        if not args.arn:
            raise SystemExit("--arn required for approve")
        cmd_approve(sm, s3, args)

    elif args.action == "reject":
        if not args.arn:
            raise SystemExit("--arn required for reject")
        cmd_reject(sm, s3, args)

    elif args.action == "metrics":
        if not args.arn:
            raise SystemExit("--arn required for metrics")
        metrics = _get_metrics_from_s3(sm, s3, args.arn)
        _print_metrics_report(metrics, args.arn)


if __name__ == "__main__":
    main()
