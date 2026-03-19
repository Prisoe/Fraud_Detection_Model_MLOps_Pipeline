"""
Lambda: formats SageMaker EventBridge events into readable SNS email alerts
for the Fraud Detection MLOps pipeline.
"""
import json
import os
from datetime import datetime, timezone

import boto3

TOPIC_ARN = os.environ["TOPIC_ARN"]
ALERTS_MODE = os.environ.get("ALERTS_MODE", "failures").lower()
PROJECT_NAME = os.environ.get("PROJECT_NAME", "fraud-detection")

sns = boto3.client("sns")


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _get(d, path, default=""):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _first(*vals):
    for v in vals:
        if v is not None and str(v).strip():
            return v
    return ""


def _format(event):
    detail_type = event.get("detail-type", "")
    region = event.get("region", "")
    account = event.get("account", "")
    time_utc = event.get("time", _utc_now())

    header = [
        f"🔍 [{PROJECT_NAME.upper()}] MLOps Alert",
        "=" * 45,
        f"Mode      : {ALERTS_MODE}",
        f"Event     : {detail_type}",
        f"Time (UTC): {time_utc}",
        f"Region    : {region}",
        f"Account   : {account}",
        "",
    ]

    # --- Pipeline execution
    if "Pipeline Execution Status Change" in detail_type:
        pipeline_arn = _first(_get(event, "detail.pipelineArn"), _get(event, "detail.PipelineArn"))
        exec_arn = _first(_get(event, "detail.pipelineExecutionArn"), _get(event, "detail.PipelineExecutionArn"))
        status = _first(_get(event, "detail.currentPipelineExecutionStatus"), _get(event, "detail.status"))
        reason = _first(_get(event, "detail.failureReason"), _get(event, "detail.FailureReason"))

        emoji = {"failed": "❌", "stopped": "🛑", "executing": "⏳", "succeeded": "✅"}.get(str(status).lower(), "ℹ️")
        subject = f"{emoji} [{PROJECT_NAME}] Pipeline {status}"
        body = header + [
            "PIPELINE EXECUTION",
            "------------------",
            f"Pipeline : {pipeline_arn}",
            f"Execution: {exec_arn}",
            f"Status   : {status}",
        ]
        if reason:
            body.append(f"Reason   : {reason}")
        return subject, "\n".join(body)

    # --- Pipeline step
    if "Pipeline Execution Step Status Change" in detail_type:
        exec_arn = _first(_get(event, "detail.pipelineExecutionArn"))
        step_name = _first(_get(event, "detail.stepName"), _get(event, "detail.StepName"))
        step_status = _first(_get(event, "detail.stepStatus"), _get(event, "detail.StepStatus"))
        reason = _first(_get(event, "detail.failureReason"))

        emoji = "❌" if str(step_status).lower() == "failed" else "⏳" if "execut" in str(step_status).lower() else "✅"
        subject = f"{emoji} [{PROJECT_NAME}] Step '{step_name}' → {step_status}"
        body = header + [
            "PIPELINE STEP",
            "-------------",
            f"Step     : {step_name}",
            f"Status   : {step_status}",
            f"Execution: {exec_arn}",
        ]
        if reason:
            body.append(f"Reason   : {reason}")
        return subject, "\n".join(body)

    # --- Model registry
    if "Model Package State Change" in detail_type:
        group = _get(event, "detail.ModelPackageGroupName")
        version = _get(event, "detail.ModelPackageVersion")
        status = _get(event, "detail.ModelPackageStatus")
        approval = _get(event, "detail.ModelApprovalStatus")
        arn = _get(event, "detail.ModelPackageArn")

        emoji = "✅" if str(approval).lower() == "approved" else "❌" if str(approval).lower() == "rejected" else "📦"
        subject = f"{emoji} [{PROJECT_NAME}] Model v{version} → {approval}"
        body = header + [
            "MODEL REGISTRY",
            "--------------",
            f"Group    : {group}",
            f"Version  : {version}",
            f"Status   : {status}",
            f"Approval : {approval}",
            f"ARN      : {arn}",
        ]
        return subject, "\n".join(body)

    # Fallback
    subject = f"ℹ️ [{PROJECT_NAME}] {detail_type}"
    body = header + ["Raw event:", json.dumps(event, indent=2)[:3000]]
    return subject, "\n".join(body)


def main(event, context):
    subject, message = _format(event)
    sns.publish(TopicArn=TOPIC_ARN, Subject=subject[:100], Message=message)
    return {"ok": True}
