"""Approve or reject a model package by ARN (Step 6 helper, kept for CLI compat)."""
import argparse, boto3, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region",           default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--model-package-arn",required=True)
    ap.add_argument("--approval-status",  choices=["Approved","Rejected","PendingManualApproval"], default="Approved")
    ap.add_argument("--approval-description", default="Approved via CLI")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    sm.update_model_package(
        ModelPackageArn=args.model_package_arn,
        ModelApprovalStatus=args.approval_status,
        ApprovalDescription=args.approval_description,
    )
    print(f"✅ {args.approval_status}: {args.model_package_arn}")

if __name__ == "__main__":
    main()
