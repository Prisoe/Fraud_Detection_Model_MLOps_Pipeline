"""
Step 9 — CloudWatch Alarm for Fraud Drift Monitoring
"""
import argparse
import boto3
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region",             default="us-east-1")
    ap.add_argument("--sns-topic-arn",      required=True)
    ap.add_argument("--namespace",          default="FraudDetection/Drift")
    ap.add_argument("--metric-name",        default="OverallPSI_Max")
    ap.add_argument("--dimension-name",     default="Project")
    ap.add_argument("--dimension-value",    default="fraud-detection")
    ap.add_argument("--threshold",          type=float, default=0.25)
    ap.add_argument("--period",             type=int,   default=300)
    ap.add_argument("--evaluation-periods", type=int,   default=1)
    ap.add_argument("--alarm-name",         default="fraud-detection-drift-alarm")
    args = ap.parse_args()

    cw = boto3.client("cloudwatch", region_name=args.region)
    cw.put_metric_alarm(
        AlarmName=args.alarm_name,
        AlarmDescription=(
            f"Triggers when feature drift (PSI) exceeds {args.threshold} "
            f"for the fraud detection model. Investigate for data shift or "
            f"new fraud patterns."
        ),
        Namespace=args.namespace,
        MetricName=args.metric_name,
        Dimensions=[{"Name": args.dimension_name, "Value": args.dimension_value}],
        Statistic="Maximum",
        Period=args.period,
        EvaluationPeriods=args.evaluation_periods,
        DatapointsToAlarm=1,
        Threshold=args.threshold,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[args.sns_topic_arn],
        OKActions=[args.sns_topic_arn],
    )
    print(f"✅ CloudWatch alarm upserted: {args.alarm_name}")
    print(f"   Namespace/Metric: {args.namespace}/{args.metric_name}")
    print(f"   Threshold: {args.threshold}")


if __name__ == "__main__":
    main()
