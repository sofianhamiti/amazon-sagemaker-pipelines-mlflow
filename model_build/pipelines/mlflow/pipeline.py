import os
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_pipeline(
        role=None,
        bucket=None,
        pipeline_name=None,
        base_job_prefix=None
):
    # parameters for pipeline execution
    tracking_uri = ParameterString(
        name='MLflowTrackingURI',
        default_value='<ADD YOUR MLFLOW LOAD BALANCER URI HERE>',
    )
    experiment_name = ParameterString(
        name='ExperimentName',
        default_value='rf-sklearn',
    )
    registered_model_name = ParameterString(
        name='RegisteredModelName',
        default_value='sklearn-random-forest',
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version='0.23-1',
        instance_type='ml.m5.xlarge',
        instance_count=1,
        base_job_name=f'{base_job_prefix}/prepare-data',
        role=role,
    )

    step_process = ProcessingStep(
        name='PrepareData',
        processor=sklearn_processor,
        code=os.path.join(BASE_DIR, 'prepare_data.py'),
        job_arguments=['--output', '/opt/ml/processing/output'],
        outputs=[
            ProcessingOutput(
                output_name='preprocessed',
                source='/opt/ml/processing/output',
                destination=bucket
            )
        ]
    )

    hyperparameters = {
        'tracking_uri': tracking_uri,
        'experiment_name': experiment_name,
        'registered_model_name': registered_model_name,
        'n-estimators': 100,
        'min-samples-leaf': 3,
        'features': 'CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT',
        'target': 'target'
    }

    metric_definitions = [{'Name': 'median-AE', 'Regex': "AE-at-50th-percentile: ([0-9.]+).*$"}]

    estimator = SKLearn(
        entry_point='train.py',
        source_dir=os.path.join(BASE_DIR, 'source_dir'),
        role=role,
        metric_definitions=metric_definitions,
        hyperparameters=hyperparameters,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        framework_version='0.23-1',
        base_job_name=f"{base_job_prefix}/train",
        disable_profiler=True
    )

    step_train = TrainingStep(
        name="TrainEvaluateRegister",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["preprocessed"].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            tracking_uri,
            experiment_name,
            registered_model_name
        ],
        steps=[step_process, step_train]
    )
    return pipeline
