import mlflow
from mlflow.tracking import MlflowClient
import tarfile
import boto3
import os
import logging
from dotenv import load_dotenv
from airflow.models import Variable
import requests

logger = logging.getLogger(__name__)


def package_and_upload_model(model_name, bucket_name, s3_key):

    load_dotenv()

     # Load MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI not set in .env")
        raise ValueError("MLFLOW_TRACKING_URI not set in .env")
    
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    try:
        model_version = client.get_model_version_by_alias(model_name, "prod")
    except Exception as e:
        raise ValueError(f"Failed to find model version with 'prod' alias for {model_name}: {e} ")
    run_id = model_version.run_id
    artifact_uri = f'runs:/{run_id}/random_forest_model'
    model_path = mlflow.artifacts.download_artifacts(artifact_uri)

    tar_path = 'model.tar.gz'
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model")
    
    s3_client = boto3.client("s3")
    
    s3_key= 'models/model.tar.gz'
    s3_client.upload_file(tar_path, bucket_name, s3_key)
    os.remove(tar_path)
    


# Define callback functions
def get_model_details(ti):
    """Retrieve MLflow model version and artifact path for the 'prod' alias."""

    load_dotenv()

    # Load MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI not set in .env")
        raise ValueError("MLFLOW_TRACKING_URI not set in .env")
    
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_version = client.get_model_version_by_alias('DeliveryDelayModelSeoul', 'prod')
    
    ti.xcom_push(key='model_version', value=model_version.version)
    ti.xcom_push(key='run_id', value=model_version.run_id)
    ti.xcom_push(key='artifact_uri', value=model_version.source)


def trigger_github_action(ti):
    try:
        model_version = ti.xcom_pull(task_ids='transition_to_production', key='production_model_version')
        model_uri = ti.xcom_pull(task_ids='transition_to_production', key='model_uri')

        model_name = 'DeliveryDelayModelSeoul'
        image_tag = f"{model_version}-2.21.3"
        logger.info(f"Triggering GitHub Actions for model {model_name}:{model_version} with MODEL_URI {model_uri}")

        github_token = Variable.get('github_token')
        github_repo = Variable.get('github_repo')
        dispatch_url = f"https://api.github.com/repos/{github_repo}/dispatches"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"
        }
        payload = {
            "event_type": "mlflow_prod_trigger",
            "client_payload": {
                "model_name": model_name,
                "model_version": model_version,
                "image_tag": image_tag,
                "model_uri": model_uri
            }
        }
        response = requests.post(dispatch_url, headers=headers, json=payload)
        if response.status_code == 204:
            logger.info(f"Successfully triggered GitHub Actions for model version {model_version}")
        else:
            logger.error(f"Failed to trigger GitHub Actions: {response.text}")
            raise Exception(f"Webhook failed: {response.text}")
    except Exception as e:
        logger.error(f"Error triggering GitHub Actions: {str(e)}")
        raise