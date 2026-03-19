"""Lambda: triggered by EventBridge schedule to auto-run the fraud pipeline."""
import os, datetime, boto3

sm = boto3.client("sagemaker")

def handler(event, context):
    pipeline = os.environ["PIPELINE_NAME"]
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    name = f"scheduled-{ts}"
    resp = sm.start_pipeline_execution(PipelineName=pipeline, PipelineExecutionDisplayName=name)
    return {"pipeline": pipeline, "executionArn": resp["PipelineExecutionArn"]}
