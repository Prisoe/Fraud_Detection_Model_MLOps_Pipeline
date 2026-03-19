import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import * as sns from "aws-cdk-lib/aws-sns";
import * as subs from "aws-cdk-lib/aws-sns-subscriptions";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as path from "path";

export class MlopsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const emailForAlerts = new cdk.CfnParameter(this, "EmailForAlerts", {
      type: "String",
      description: "Email for fraud-model MLOps alerts.",
    });
    const alertsMode = new cdk.CfnParameter(this, "AlertsMode", {
      type: "String",
      allowedValues: ["failures", "all"],
      default: "failures",
    });

    // ── S3: encrypted, versioned, private
    const artifactsBucket = new s3.Bucket(this, "FraudArtifactsBucket", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // ── SageMaker IAM role — SCOPED DOWN (v2 upgrade)
    // Replaces AmazonSageMakerFullAccess with least-privilege permissions
    const sagemakerRole = new iam.Role(this, "FraudSageMakerRole", {
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
      description: "Scoped SageMaker execution role for fraud detection pipeline",
    });

    // S3: only the artifacts bucket
    artifactsBucket.grantReadWrite(sagemakerRole);

    // SageMaker: only the actions actually needed
    sagemakerRole.addToPolicy(new iam.PolicyStatement({
      sid: "SageMakerPipelineOperations",
      effect: iam.Effect.ALLOW,
      actions: [
        // Processing jobs
        "sagemaker:CreateProcessingJob",   "sagemaker:DescribeProcessingJob",
        "sagemaker:StopProcessingJob",     "sagemaker:ListProcessingJobs",
        // Training jobs
        "sagemaker:CreateTrainingJob",     "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob",       "sagemaker:ListTrainingJobs",
        // Models
        "sagemaker:CreateModel",           "sagemaker:DescribeModel",
        "sagemaker:DeleteModel",           "sagemaker:ListModels",
        // Endpoint configs + endpoints
        "sagemaker:CreateEndpointConfig",  "sagemaker:DescribeEndpointConfig",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:CreateEndpoint",        "sagemaker:DescribeEndpoint",
        "sagemaker:UpdateEndpoint",        "sagemaker:DeleteEndpoint",
        "sagemaker:InvokeEndpoint",
        // Model Registry
        "sagemaker:CreateModelPackage",    "sagemaker:DescribeModelPackage",
        "sagemaker:UpdateModelPackage",    "sagemaker:ListModelPackages",
        "sagemaker:CreateModelPackageGroup",
        "sagemaker:DescribeModelPackageGroup",
        "sagemaker:ListModelPackageGroups",
        // Pipeline execution
        "sagemaker:CreatePipeline",        "sagemaker:DescribePipeline",
        "sagemaker:UpdatePipeline",        "sagemaker:DeletePipeline",
        "sagemaker:StartPipelineExecution",
        "sagemaker:StopPipelineExecution",
        "sagemaker:DescribePipelineExecution",
        "sagemaker:ListPipelineExecutionSteps",
        "sagemaker:ListPipelineExecutions",
        "sagemaker:ListPipelines",
        // Tagging — REQUIRED: SageMaker Pipelines tags every job it creates
        "sagemaker:AddTags",
        "sagemaker:ListTags",
        "sagemaker:DeleteTags",
        // Experiments / search (used internally by Pipelines)
        "sagemaker:CreateExperiment",
        "sagemaker:CreateTrial",
        "sagemaker:CreateTrialComponent",
        "sagemaker:UpdateTrialComponent",
        "sagemaker:AssociateTrialComponent",
        "sagemaker:DescribeTrialComponent",
        "sagemaker:Search",
      ],
      resources: ["*"],
    }));

    // ECR: pull training/processing container images
    sagemakerRole.addToPolicy(new iam.PolicyStatement({
      sid: "ECRPull",
      effect: iam.Effect.ALLOW,
      actions: [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
      ],
      resources: ["*"],
    }));

    // CloudWatch Logs: write training/processing job logs
    sagemakerRole.addToPolicy(new iam.PolicyStatement({
      sid: "CloudWatchLogs",
      effect: iam.Effect.ALLOW,
      actions: [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
      ],
      resources: [`arn:aws:logs:*:*:log-group:/aws/sagemaker/*`],
    }));

    // CloudWatch Metrics: publish drift metrics
    sagemakerRole.addToPolicy(new iam.PolicyStatement({
      sid: "CloudWatchMetrics",
      effect: iam.Effect.ALLOW,
      actions: ["cloudwatch:PutMetricData"],
      resources: ["*"],
    }));

    // IAM PassRole: SageMaker needs to pass this role to jobs
    sagemakerRole.addToPolicy(new iam.PolicyStatement({
      sid: "PassRole",
      effect: iam.Effect.ALLOW,
      actions: ["iam:PassRole"],
      resources: ["*"],
      conditions: {
        StringEquals: { "iam:PassedToService": "sagemaker.amazonaws.com" },
      },
    }));

    // ── SNS alerts
    const alertsTopic = new sns.Topic(this, "FraudAlertsTopic", {
      displayName: "Fraud Detection MLOps Alerts",
    });
    alertsTopic.addSubscription(new subs.EmailSubscription(emailForAlerts.valueAsString));

    // ── Lambda: EventBridge → formatted SNS email
    const formatterFn = new lambda.Function(this, "FraudAlertsFormatter", {
      runtime: lambda.Runtime.PYTHON_3_10,
      handler: "handler.main",
      code: lambda.Code.fromAsset(path.join(__dirname, "../lambda/alerts_formatter")),
      timeout: cdk.Duration.seconds(15),
      environment: {
        TOPIC_ARN:    alertsTopic.topicArn,
        ALERTS_MODE:  alertsMode.valueAsString,
        PROJECT_NAME: "fraud-detection",
      },
    });
    alertsTopic.grantPublish(formatterFn);

    const smSource = ["aws.sagemaker"];

    // Pipeline execution: all (test mode)
    new events.Rule(this, "FraudPipelineAllRule", {
      enabled: alertsMode.valueAsString === "all",
      eventPattern: { source: smSource, detailType: ["SageMaker Model Building Pipeline Execution Status Change"] },
    }).addTarget(new targets.LambdaFunction(formatterFn));

    // Pipeline execution: failures always
    new events.Rule(this, "FraudPipelineFailedRule", {
      eventPattern: {
        source: smSource,
        detailType: ["SageMaker Model Building Pipeline Execution Status Change"],
        detail: { currentPipelineExecutionStatus: ["Failed"] },
      },
    }).addTarget(new targets.LambdaFunction(formatterFn));

    // Step failures always
    new events.Rule(this, "FraudStepFailedRule", {
      eventPattern: {
        source: smSource,
        detailType: ["SageMaker Model Building Pipeline Execution Step Status Change"],
        detail: { stepStatus: ["Failed"] },
      },
    }).addTarget(new targets.LambdaFunction(formatterFn));

    // All step changes (test mode)
    new events.Rule(this, "FraudStepAllRule", {
      enabled: alertsMode.valueAsString === "all",
      eventPattern: { source: smSource, detailType: ["SageMaker Model Building Pipeline Execution Step Status Change"] },
    }).addTarget(new targets.LambdaFunction(formatterFn));

    // Model registry: always on
    new events.Rule(this, "FraudModelPackageRule", {
      eventPattern: { source: smSource, detailType: ["SageMaker Model Package State Change"] },
    }).addTarget(new targets.LambdaFunction(formatterFn));

    // ── Outputs
    new cdk.CfnOutput(this, "ArtifactsBucketName",  { value: artifactsBucket.bucketName, exportName: "FraudArtifactsBucketName" });
    new cdk.CfnOutput(this, "SageMakerRoleArn",     { value: sagemakerRole.roleArn,      exportName: "FraudSageMakerRoleArn" });
    new cdk.CfnOutput(this, "AlertsTopicArn",       { value: alertsTopic.topicArn,       exportName: "FraudAlertsTopicArn" });
  }
}
