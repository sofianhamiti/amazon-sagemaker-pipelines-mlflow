import os
import logging
import mlflow
import tarfile
import sagemaker
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


class MLflowHandler:
    def __init__(self, tracking_uri, model_name, model_version):
        mlflow.set_registry_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
        self.model_version = model_version
        self.model_stage = None
        logging.info('MLFLOW HANDLER LOADED')

    def _download_model_version_files(self):
        """
        download model version files to a local tmp folder.
        """
        model_version = self.client.get_model_version(name=self.model_name, version=self.model_version)
        artifact_uri = model_version.source
        return _download_artifact_from_uri(artifact_uri)

    @staticmethod
    def _make_tar_gz_file(output_filename, source_dir):
        """
        create a tar.gz from a directory.
        """
        with tarfile.open(output_filename, "w:gz") as tar:
            for f in os.listdir(source_dir):
                tar.add(os.path.join(source_dir, f), arcname=f)

    def prepare_sagemaker_model(self):
        """
        create and upload a tar.gz to S3 from a chosen MLflow model version.
        """
        try:
            sagemaker_session = sagemaker.Session()
            bucket = sagemaker_session.default_bucket()  # you can specify other bucket name here
            prefix = f'mlflow_model/{self.model_name}-{self.model_version}'
            tmp_file = '/tmp/model.tar.gz'

            model_local_path = self._download_model_version_files()
            self._make_tar_gz_file(tmp_file, model_local_path)
            tar_gz_s3_location = sagemaker_session.upload_data(path=tmp_file, bucket=bucket, key_prefix=prefix)
            logging.info(f'model.tar.gz upload to {tar_gz_s3_location}')

            return tar_gz_s3_location

        except Exception as e:
            logging.error(e)

    def transition_model_version_stage(self, stage):
        """
        Transitions a model version to input stage.
        Transitions other model versions to Archived if they were in Staging or Production.
        """
        try:
            for model in self.client.search_model_versions(f"name='{self.model_name}'"):
                if model.current_stage in ['Staging', 'Production']:
                    self.client.transition_model_version_stage(
                        name=model.name,
                        version=model.version,
                        stage="Archived"
                    )
                    logging.info(f'Transitioning {model.name}/{model.version} to Archived')

            self.client.transition_model_version_stage(
                name=self.model_name,
                version=self.model_version,
                stage=stage
            )
            logging.info(f'Model transitioned to {stage}')

        except Exception as e:
            logging.error(e)
