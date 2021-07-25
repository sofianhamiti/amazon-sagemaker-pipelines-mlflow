import os
import json
import logging
import argparse
from mlflow_handler import MLflowHandler

logger = logging.getLogger(__name__)


def extend_config(args, model_data_location, stage_config):
    """
    Extend the stage configuration with additional parameters and tags based.
    """
    # Verify that config has parameters and tags sections
    if not "Parameters" in stage_config or not "StageName" in stage_config["Parameters"]:
        raise Exception("Configuration file must include StageName parameter")
    if not "Tags" in stage_config:
        stage_config["Tags"] = {}
    # Create new params and tags
    new_params = {
        "SageMakerProjectName": args.sagemaker_project_name,
        "ModelDataLocation": model_data_location,
        "ContainerImageURI": args.container_image_uri,
        "ModelExecutionRoleArn": args.model_execution_role,
        "EndpointInstanceCount": args.initial_instance_count,
        "EndpointInstanceType": args.instance_type
    }
    new_tags = {
        "sagemaker:deployment-stage": stage_config["Parameters"]["StageName"],
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
        "ModelName": args.model_name,
        "ModelVersion": args.model_version,
        "TrackingURI": args.tracking_uri
    }
    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--import-prod-config", type=str, default="prod-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-prod-config", type=str, default="prod-config-export.json")
    parser.add_argument("--tracking-uri", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--container-image-uri", type=str, required=True)
    parser.add_argument("--initial-instance-count", type=str, required=True)
    parser.add_argument("--instance-type", type=str, required=True)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    mlflow_handler = MLflowHandler(
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        model_version=args.model_version
    )
    # create and upload a tar.gz to S3 from the MLflow model version.
    model_data_location = mlflow_handler.prepare_sagemaker_model()

    # Write the staging config
    with open(args.import_staging_config, "r") as f:
        staging_config = extend_config(args, model_data_location, json.load(f))
    logger.debug("Staging config: {}".format(json.dumps(staging_config, indent=4)))
    with open(args.export_staging_config, "w") as f:
        json.dump(staging_config, f, indent=4)

    # Write the prod config
    with open(args.import_prod_config, "r") as f:
        prod_config = extend_config(args, model_data_location, json.load(f))
    logger.debug("Prod config: {}".format(json.dumps(prod_config, indent=4)))
    with open(args.export_prod_config, "w") as f:
        json.dump(prod_config, f, indent=4)

    # Transition model version to Staging
    mlflow_handler.transition_model_version_stage('Staging')
