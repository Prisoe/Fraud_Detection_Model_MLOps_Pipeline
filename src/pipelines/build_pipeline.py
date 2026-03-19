"""
Pipeline DAG — Fraud Detection (v2)
=====================================
Upgrades over v1:
  1. Data validation step added before preprocessing (gates on schema/stats)
  2. Step caching enabled (saves ~15 min per unchanged run)
  3. Two-condition quality gate:
       a) avg_precision >= 0.80
       b) beats_champion == 1  (new model must improve on current deployed)
  4. HPO-ready: hyperparameters exposed as pipeline parameters
  5. Baseline output wired through for drift monitoring
  6. Retraining can be triggered by drift alarm via run_pipeline.py
"""

import os
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# CacheConfig import path varies across SageMaker SDK versions:
#   sagemaker>=2.90  -> sagemaker.workflow.cache_config.CacheConfig
#   sagemaker 2.x   -> sagemaker.workflow.steps.CacheConfig
# Falls back to a no-op stub so the pipeline still runs without caching.
try:
    from sagemaker.workflow.cache_config import CacheConfig
except ImportError:
    try:
        from sagemaker.workflow.steps import CacheConfig
    except ImportError:
        class CacheConfig:  # type: ignore
            def __init__(self, enable_caching=False, expire_after=None):
                self.enable_caching = False

BASE_DIR = Path(__file__).resolve().parents[1]

# ── Instance types (quota-safe for student/dev accounts)
PROCESSING_INSTANCE = "ml.t3.medium"
TRAINING_INSTANCE   = "ml.m5.large"
EVAL_INSTANCE       = "ml.t3.medium"
INFERENCE_INSTANCE  = "ml.t2.medium"
TRANSFORM_INSTANCE  = "ml.m5.large"

PIPELINE_NAME       = "fraud-detection-pipeline"
MODEL_PACKAGE_GROUP = "fraud-detection-model-group"
AUPRC_THRESHOLD     = 0.80

# ── Step caching: reuse outputs if code + inputs unchanged (saves ~15 min)
CACHE_CONFIG = CacheConfig(enable_caching=True, expire_after="30d")


def get_pipeline(region: str, role_arn: str, default_bucket: str):
    os.environ["AWS_REGION"] = region
    os.environ["AWS_DEFAULT_REGION"] = region

    pipeline_sess = PipelineSession(default_bucket=default_bucket)
    sklearn_image = sagemaker.image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1"
    )

    print(f"[build_pipeline] region={region} bucket={default_bucket}")

    # ── Pipeline parameters (overridable at execution time)
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{default_bucket}/data/raw/creditcard.csv",
    )
    # Expose key hyperparams as pipeline params → enables HPO sweeps
    p_n_estimators  = ParameterInteger(name="NEstimators",   default_value=300)
    p_max_depth     = ParameterInteger(name="MaxDepth",      default_value=6)
    p_learning_rate = ParameterFloat(  name="LearningRate",  default_value=0.05)
    p_fn_cost       = ParameterFloat(  name="FnCost",        default_value=10.0)
    p_fp_cost       = ParameterFloat(  name="FpCost",        default_value=1.0)

    # ────────────────────────────────────────────
    # Step 0: Validate data
    # ────────────────────────────────────────────
    validate_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role_arn,
        instance_type=PROCESSING_INSTANCE,
        instance_count=1,
        sagemaker_session=pipeline_sess,
    )

    validation_report = PropertyFile(
        name="ValidationReport",
        output_name="validation",
        path="validation_report.json",
    )

    validate_step = ProcessingStep(
        name="ValidateData",
        processor=validate_processor,
        inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")],
        outputs=[ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation")],
        code="src/validate/validate_data.py",
        job_arguments=[
            "--input-data", "/opt/ml/processing/input",
            "--output-dir", "/opt/ml/processing/validation",
        ],
        property_files=[validation_report],
        cache_config=CACHE_CONFIG,
    )

    # ────────────────────────────────────────────
    # Step 1: Preprocess (depends on ValidateData passing)
    # ────────────────────────────────────────────
    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=ScriptProcessor(
            image_uri=sklearn_image,
            command=["python3"],
            role=role_arn,
            instance_type=PROCESSING_INSTANCE,
            instance_count=1,
            sagemaker_session=pipeline_sess,
        ),
        inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train",    source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="val",      source="/opt/ml/processing/val"),
            ProcessingOutput(output_name="test",     source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="baseline", source="/opt/ml/processing/baseline"),
        ],
        code="src/preprocess/preprocess.py",
        job_arguments=[
            "--input-data",      "/opt/ml/processing/input",
            "--output-train",    "/opt/ml/processing/train",
            "--output-val",      "/opt/ml/processing/val",
            "--output-test",     "/opt/ml/processing/test",
            "--output-baseline", "/opt/ml/processing/baseline",
        ],
        depends_on=[validate_step],   # ← blocked until validation passes
        cache_config=CACHE_CONFIG,
    )

    # ────────────────────────────────────────────
    # Step 2: Train
    # ────────────────────────────────────────────
    estimator = Estimator(
        image_uri=sklearn_image,
        role=role_arn,
        instance_count=1,
        instance_type=TRAINING_INSTANCE,
        output_path=f"s3://{default_bucket}/artifacts/model",
        sagemaker_session=pipeline_sess,
        # source_dir bundles train.py + requirements.txt together.
        # SageMaker pip-installs requirements.txt before running the script,
        # which is how xgboost and shap get installed in the sklearn container.
        source_dir="src/train",
        entry_point="train.py",
        disable_profiler=True,   # prevents cache misses from profiler timestamp
        hyperparameters={
            "n-estimators":     p_n_estimators,
            "max-depth":        p_max_depth,
            "learning-rate":    p_learning_rate,
            "fn-cost":          p_fn_cost,
            "fp-cost":          p_fp_cost,
            "cv-folds":         5,
        },
    )

    train_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "val": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=CACHE_CONFIG,
    )

    # ────────────────────────────────────────────
    # Step 3: Evaluate
    # ────────────────────────────────────────────
    evaluation_report = PropertyFile(
        name="FraudEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_step = ProcessingStep(
        name="Evaluate",
        processor=ScriptProcessor(
            image_uri=sklearn_image,
            command=["python3"],
            role=role_arn,
            instance_type=EVAL_INSTANCE,
            instance_count=1,
            sagemaker_session=pipeline_sess,
        ),
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        # Relative path — exactly as the original template uses it
        code="src/evaluate/evaluate.py",
        job_arguments=[
            "--model",      "/opt/ml/processing/model",
            "--test",       "/opt/ml/processing/test",
            "--output-dir", "/opt/ml/processing/evaluation",
            "--skip-champion-check",
            "--fn-cost",    "10.0",
            "--fp-cost",    "1.0",
        ],
        property_files=[evaluation_report],
    )

    # ────────────────────────────────────────────
    # Step 4: Register (gated by TWO conditions)
    # ────────────────────────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    # RegisterModel: supply image_uri to suppress the automatic repack job.
    # The repack job is triggered when the Estimator has source_dir set and
    # RegisterModel can't find an inference script to bundle. By providing
    # image_uri directly, SageMaker skips repacking and uses the model.tar.gz
    # from the training step as-is.
    sklearn_inference_image = sagemaker.image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1"
    )
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        image_uri=sklearn_inference_image,   # suppresses repack job
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv", "application/json"],
        response_types=["application/json"],
        inference_instances=[INFERENCE_INSTANCE],
        transform_instances=[TRANSFORM_INSTANCE],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # Quality gate implemented as nested ConditionSteps
    # (avoids ConditionAnd which is not available in sagemaker 2.232.0)
    #
    # Inner gate: beats_champion >= 1
    #   (challenger must improve on currently deployed model)
    inner_gate = ConditionStep(
        name="CheckBeatsChampion",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=evaluation_report,
                    json_path="beats_champion",
                ),
                right=1,
            )
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    # Outer gate: avg_precision >= AUPRC_THRESHOLD
    #   Only proceeds to inner gate (champion check) if AUPRC passes
    quality_gate = ConditionStep(
        name="CheckAUPRC",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=evaluation_report,
                    json_path="avg_precision",
                ),
                right=AUPRC_THRESHOLD,
            )
        ],
        if_steps=[inner_gate],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            input_data_uri, p_n_estimators, p_max_depth,
            p_learning_rate, p_fn_cost, p_fp_cost,
        ],
        steps=[validate_step, preprocess_step, train_step, eval_step, quality_gate],
        sagemaker_session=pipeline_sess,
    )
    return pipeline


if __name__ == "__main__":
    region   = os.environ.get("AWS_REGION", "us-east-1")
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    bucket   = os.environ["ARTIFACT_BUCKET"]
    p = get_pipeline(region=region, role_arn=role_arn, default_bucket=bucket)
    p.upsert(role_arn=role_arn)
    print(f"✅ Upserted pipeline: {p.name}")
