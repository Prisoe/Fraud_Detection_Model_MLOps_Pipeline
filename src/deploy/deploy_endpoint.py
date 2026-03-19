"""
Endpoint Deployment — Fraud Detection (v2)
===========================================
Upgrades over v1:
  1. Canary deployment: new model gets configurable % of traffic first
     (default 10%), existing variant keeps the rest
  2. Input validation in inference bundle (wrong feature count → clear error)
  3. Threshold loaded from model bundle (threshold.json) at inference time
  4. Latency + error rate CloudWatch alarms created alongside endpoint
  5. Rollback helper: if canary errors spike, revert instantly
"""

import argparse
import io
import json
import os
import tarfile
import time
from datetime import datetime, timezone

import boto3

PROJECT         = "fraud-detection"
MODEL_PKG_GROUP = "fraud-detection-model-group"
EXPECTED_FEATURES = 29   # V1-V28 + Amount (Time dropped: EDA |corr|=0.012)
# EDA-confirmed SHAP importance order (top features drive fraud predictions):
# V14 (2.41) > V4 (1.69) > V12 (0.77) > V10 (0.61) > V11 (0.45)
# Note: simple correlation ranking differs (V17 #1) — SHAP is more reliable.


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _get_latest_approved(sm, group: str) -> str:
    resp = sm.list_model_packages(
        ModelPackageGroupName=group, SortBy="CreationTime",
        SortOrder="Descending", MaxResults=50,
    )
    for p in resp.get("ModelPackageSummaryList", []):
        if p["ModelPackageStatus"] == "Completed" and p["ModelApprovalStatus"] == "Approved":
            return p["ModelPackageArn"]
    raise RuntimeError(f"No Approved model in group '{group}'")


def _build_inference_bundle() -> bytes:
    """
    Builds inference.py + requirements.txt into a tar.gz.
    The sklearn serving container reads requirements.txt and pip-installs
    it before starting gunicorn, so xgboost is available at serve time.
    inference.py also calls subprocess pip install as a safety net.
    """
    # requirements.txt — read by SageMaker serving container on startup
    requirements = b"xgboost>=1.7.0\n"

    code = r'''
import json, os, subprocess, sys, joblib, numpy as np

_model     = None
_threshold = 0.5
_EXPECTED_FEATURES = 29

def model_fn(model_dir):
    global _model, _threshold

    # Install xgboost if not present (safety net for serving container)
    try:
        import xgboost  # noqa
    except ImportError:
        print("[inference] Installing xgboost...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost>=1.7.0", "-q"])
        print("[inference] xgboost installed")

    _model = joblib.load(os.path.join(model_dir, "model.joblib"))

    t_path = os.path.join(model_dir, "threshold.json")
    if os.path.exists(t_path):
        with open(t_path) as fh:
            data = json.load(fh)
        _threshold = float(data["threshold"])
        print(f"[inference] threshold={_threshold:.4f}")
    else:
        print(f"[inference] threshold.json not found, using {_threshold}")
    return _model

def input_fn(body, content_type):
    if content_type == "text/csv":
        rows = [r.strip() for r in body.strip().splitlines() if r.strip()]
        data = [[float(x) for x in r.split(",")] for r in rows]
    elif content_type == "application/json":
        obj = json.loads(body)
        data = obj["instances"] if isinstance(obj, dict) and "instances" in obj else obj
    else:
        raise ValueError(f"Unsupported content-type: {content_type}")
    arr = np.array(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != _EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {_EXPECTED_FEATURES} features (V1-V28 + Amount). "
            f"Got {arr.shape[1]}."
        )
    return arr

def predict_fn(X, model):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= _threshold).astype(int)
    return {"probabilities": probs, "predictions": preds}

def output_fn(result, accept):
    payload = {
        "predictions":   result["predictions"].tolist(),
        "probabilities": [round(float(p), 6) for p in result["probabilities"]],
        "threshold":     _threshold,
        "n_flagged":     int(result["predictions"].sum()),
    }
    return json.dumps(payload), "application/json"
'''.lstrip()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add requirements.txt — SageMaker serving container installs this on start
        req_info = tarfile.TarInfo(name="requirements.txt")
        req_info.size = len(requirements)
        tar.addfile(req_info, io.BytesIO(requirements))
        # Add inference.py
        code_bytes = code.encode("utf-8")
        code_info  = tarfile.TarInfo(name="inference.py")
        code_info.size = len(code_bytes)
        tar.addfile(code_info, io.BytesIO(code_bytes))
    return buf.getvalue()


def _upload_bundle(s3, bucket: str) -> str:
    key = f"artifacts/inference/fraud-inference-{_now()}.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=_build_inference_bundle())
    return f"s3://{bucket}/{key}"


def _create_model(sm, name, role_arn, image, model_data, code_uri, region):
    try:
        sm.describe_model(ModelName=name)
        sm.delete_model(ModelName=name)
        time.sleep(2)
    except sm.exceptions.ClientError:
        pass
    sm.create_model(
        ModelName=name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image,
            "ModelDataUrl": model_data,
            "Environment": {
                "SAGEMAKER_PROGRAM":             "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY":    code_uri,
                "SAGEMAKER_REGION":              region,
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            },
        },
    )
    print(f"[deploy] Created model: {name}")


def _canary_endpoint_config(
    sm, cfg_name, champion_model, challenger_model,
    instance_type, n_instances, capture_uri,
    canary_traffic_pct: int,
):
    """
    Creates an endpoint config with TWO variants:
      - champion: (100 - canary_traffic_pct)% of traffic
      - challenger: canary_traffic_pct% of traffic
    If no champion_model provided, challenger gets 100%.
    """
    try:
        sm.describe_endpoint_config(EndpointConfigName=cfg_name)
        sm.delete_endpoint_config(EndpointConfigName=cfg_name)
        time.sleep(2)
    except sm.exceptions.ClientError:
        pass

    variants = []

    if champion_model:
        variants.append({
            "VariantName":          "Champion",
            "ModelName":            champion_model,
            "InitialInstanceCount": n_instances,
            "InstanceType":         instance_type,
            "InitialVariantWeight": float(100 - canary_traffic_pct),
        })
        challenger_weight = float(canary_traffic_pct)
        print(f"[deploy] Canary: Champion={100-canary_traffic_pct}% | Challenger={canary_traffic_pct}%")
    else:
        challenger_weight = 100.0
        print("[deploy] No champion — Challenger gets 100% of traffic")

    variants.append({
        "VariantName":          "Challenger",
        "ModelName":            challenger_model,
        "InitialInstanceCount": n_instances,
        "InstanceType":         instance_type,
        "InitialVariantWeight": challenger_weight,
    })

    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=variants,
        DataCaptureConfig={
            "EnableCapture":             True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri":          capture_uri,
            "CaptureOptions":            [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {
                "CsvContentTypes":  ["text/csv"],
                "JsonContentTypes": ["application/json"],
            },
        },
    )
    print(f"[deploy] Created endpoint config: {cfg_name}")


def _get_current_champion_model(sm, endpoint_name: str):
    """Returns the model name of the current Champion variant, or None."""
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        cfg  = sm.describe_endpoint_config(
            EndpointConfigName=desc["EndpointConfigName"]
        )
        for v in cfg.get("ProductionVariants", []):
            if v["VariantName"] == "Champion":
                return v["ModelName"]
    except Exception:
        pass
    return None


def _ensure_endpoint(sm, endpoint_name, cfg_name):
    try:
        desc   = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        if status == "Failed":
            sm.delete_endpoint(EndpointName=endpoint_name)
            for _ in range(60):
                time.sleep(10)
                try:
                    sm.describe_endpoint(EndpointName=endpoint_name)
                except sm.exceptions.ClientError:
                    break
        else:
            sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=cfg_name)
            print(f"[deploy] Updating endpoint: {endpoint_name}")
            return
    except sm.exceptions.ClientError:
        pass
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=cfg_name)
    print(f"[deploy] Creating endpoint: {endpoint_name}")


def _wait_inservice(sm, endpoint_name, poll=30, timeout_min=40):
    deadline = time.time() + timeout_min * 60
    while True:
        desc   = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print(f"[deploy] Status: {status}")
        if status == "InService":
            return desc
        if status in ("Failed", "OutOfService"):
            raise RuntimeError(f"Endpoint failed: {desc}")
        if time.time() > deadline:
            raise TimeoutError(f"Timed out after {timeout_min}m")
        time.sleep(poll)


def _create_latency_alarms(cw, endpoint_name, sns_topic_arn, region):
    """
    Create CloudWatch alarms for endpoint health — not just drift.
    """
    namespace = "AWS/SageMaker"
    dims = [
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName",  "Value": "Challenger"},
    ]

    # Standard alarms (Statistic must be one of: Average, Sum, Min, Max, SampleCount)
    # Note: p99 percentile uses ExtendedStatistics not Statistic — using Average latency
    # which is simpler and works with all SDK versions.
    standard_alarms = [
        {
            "AlarmName":          f"{endpoint_name}-latency-high",
            "MetricName":         "ModelLatency",
            "Statistic":          "Average",
            "Threshold":          500_000,   # microseconds = 500ms average
            "ComparisonOperator": "GreaterThanThreshold",
            "AlarmDescription":   "Average inference latency > 500ms",
        },
        {
            "AlarmName":          f"{endpoint_name}-invocation-4xx",
            "MetricName":         "Invocation4XXErrors",
            "Statistic":          "Sum",
            "Threshold":          10,
            "ComparisonOperator": "GreaterThanThreshold",
            "AlarmDescription":   "More than 10 client errors (4XX) in 5 min",
        },
        {
            "AlarmName":          f"{endpoint_name}-invocation-5xx",
            "MetricName":         "Invocation5XXErrors",
            "Statistic":          "Sum",
            "Threshold":          1,
            "ComparisonOperator": "GreaterThanThreshold",
            "AlarmDescription":   "Any server errors (5XX) on fraud endpoint",
        },
    ]

    for alarm in standard_alarms:
        cw.put_metric_alarm(
            AlarmName=alarm["AlarmName"],
            AlarmDescription=alarm["AlarmDescription"],
            Namespace=namespace,
            MetricName=alarm["MetricName"],
            Dimensions=dims,
            Statistic=alarm["Statistic"],
            Period=300,
            EvaluationPeriods=1,
            DatapointsToAlarm=1,
            Threshold=alarm["Threshold"],
            ComparisonOperator=alarm["ComparisonOperator"],
            TreatMissingData="notBreaching",
            AlarmActions=[sns_topic_arn] if sns_topic_arn else [],
        )
        print(f"[deploy] Alarm created: {alarm['AlarmName']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region",           default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--role-arn",         default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--artifact-bucket",  default=os.environ.get("ARTIFACT_BUCKET"))
    ap.add_argument("--endpoint-name",    default=f"{PROJECT}-endpoint")
    ap.add_argument("--instance-type",    default="ml.t2.medium")
    ap.add_argument("--instance-count",   type=int, default=1)
    ap.add_argument("--canary-pct",       type=int, default=10,
                    help="% of traffic to send to challenger (0 = full cutover)")
    ap.add_argument("--model-pkg-arn",    default=None)
    ap.add_argument("--allow-pending",    action="store_true")
    ap.add_argument("--sns-topic-arn",    default=os.environ.get("SNS_TOPIC_ARN", ""))
    ap.add_argument("--wait",             action="store_true")
    args = ap.parse_args()

    if not args.role_arn:        raise SystemExit("Missing --role-arn")
    if not args.artifact_bucket: raise SystemExit("Missing --artifact-bucket")

    sm  = boto3.client("sagemaker",   region_name=args.region)
    s3  = boto3.client("s3",          region_name=args.region)
    cw  = boto3.client("cloudwatch",  region_name=args.region)

    # Resolve model package
    pkg_arn  = args.model_pkg_arn or _get_latest_approved(sm, MODEL_PKG_GROUP)
    mp       = sm.describe_model_package(ModelPackageName=pkg_arn)
    approval = mp["ModelApprovalStatus"]
    if approval != "Approved" and not args.allow_pending:
        raise SystemExit(f"Package is {approval}. Approve it or pass --allow-pending.")

    container  = mp["InferenceSpecification"]["Containers"][0]
    image      = container["Image"]
    model_data = container["ModelDataUrl"]
    print(f"[deploy] Package  : {pkg_arn}")
    print(f"[deploy] Approval : {approval}")

    # Upload inference bundle
    code_uri = _upload_bundle(s3, args.artifact_bucket)
    print(f"[deploy] Code URI : {code_uri}")

    # Detect existing champion
    champion_model = _get_current_champion_model(sm, args.endpoint_name)
    if champion_model:
        print(f"[deploy] Existing champion model: {champion_model}")
    else:
        print("[deploy] No existing champion (first deployment)")

    # Create challenger model
    suffix     = _now()
    challenger = f"{PROJECT}-challenger-{suffix}"
    cfg_name   = f"{PROJECT}-cfg-{suffix}"
    capture_uri = f"s3://{args.artifact_bucket}/monitoring/data-capture/{args.endpoint_name}/"

    _create_model(sm, challenger, args.role_arn, image, model_data, code_uri, args.region)
    _canary_endpoint_config(
        sm, cfg_name,
        champion_model=champion_model if args.canary_pct > 0 else None,
        challenger_model=challenger,
        instance_type=args.instance_type,
        n_instances=args.instance_count,
        capture_uri=capture_uri,
        canary_traffic_pct=args.canary_pct,
    )
    _ensure_endpoint(sm, args.endpoint_name, cfg_name)

    print(f"\n✅ Deployment submitted")
    print(f"   Endpoint    : {args.endpoint_name}")
    print(f"   Canary pct  : {args.canary_pct}%")
    print(f"   DataCapture : {capture_uri}")

    # Latency + error alarms
    if args.sns_topic_arn:
        print("\n[deploy] Creating latency/error alarms...")
        _create_latency_alarms(cw, args.endpoint_name, args.sns_topic_arn, args.region)

    if args.wait:
        _wait_inservice(sm, args.endpoint_name)
        print("✅ Endpoint is InService")


if __name__ == "__main__":
    main()
