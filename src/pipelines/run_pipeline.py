"""Trigger a pipeline execution (used by deploy scripts and Lambda scheduler)."""
import os
import boto3


def main():
    region = os.environ.get("AWS_REGION", "us-east-1")
    sm = boto3.client("sagemaker", region_name=region)
    pipeline_name = "fraud-detection-pipeline"

    resp = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName="manual-run",
    )
    print("✅ Started execution:", resp["PipelineExecutionArn"])


if __name__ == "__main__":
    main()
