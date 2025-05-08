import boto3
import time

sm = boto3.client("sagemaker", region_name="ap-southeast-2")

MODEL_NAME = "Financial"
ENDPOINT_NAME = "Financial-endpoint"
ROLE_ARN = "arn:aws:iam::620437283788:role/MLOPs"
MODEL_ARTIFACT = "s3://modelmlops/Model/model.tar.gz"
INSTANCE_TYPE = "ml.t2.medium"

# ----- 1. Create Model -----
print("Creating SageMaker model...")
try:
    sm.create_model(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            "Image": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py38-ubuntu20.04",  # HF Container
            "ModelDataUrl": MODEL_ARTIFACT,
            "Environment": {
                "HF_TASK": "text-generation"  # ใช้กับ LLaMA, GPT, etc.
            }
        }
    )
except sm.exceptions.ClientError as e:
    if "already exists" in str(e):
        print("⚠️ Model already exists.")
    else:
        raise

# ----- 2. Create Endpoint Config -----
print("Creating endpoint config...")
config_name = f"{ENDPOINT_NAME}-config"
try:
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1
            }
        ]
    )
except sm.exceptions.ClientError as e:
    if "already exists" in str(e):
        print("Endpoint config already exists.")
    else:
        raise

# ----- 3. Create Endpoint (or update if exists) -----
existing_endpoints = sm.list_endpoints(NameContains=ENDPOINT_NAME)["Endpoints"]
if existing_endpoints:
    print("Updating existing endpoint...")
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name
    )
else:
    print("Creating new endpoint...")
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name
    )

print("Waiting for endpoint to be InService...")
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=ENDPOINT_NAME)
print("Endpoint ready:", ENDPOINT_NAME)
