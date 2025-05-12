# import boto3
# import time
# import json
# import os

# # ---------- Load Config ----------
# CONFIG_PATH = "configs/sagemaker_config.json"
# if not os.path.exists(CONFIG_PATH):
#     raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

# with open(CONFIG_PATH) as f:
#     cfg = json.load(f)

# MODEL_NAME = cfg["model_name"]
# ENDPOINT_NAME = cfg["endpoint_name"]
# ROLE_ARN = cfg["role_arn"]
# MODEL_ARTIFACT = cfg["model_artifact"]  # s3://...
# INSTANCE_TYPE = cfg.get("instance_type", "ml.t3.medium")
# REGION = cfg.get("region", "ap-southeast-2")
# IMAGE_URI = cfg.get("image_uri", "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py38-ubuntu20.04")

# sm = boto3.client("sagemaker", region_name=REGION)

# # ---------- Clean Up Existing Model and Config ----------
# print("Deleting existing model and endpoint config if exist...")
# for res_type, name in [("model", MODEL_NAME), ("endpoint-config", f"{ENDPOINT_NAME}-config")]:
#     try:
#         if res_type == "model":
#             sm.delete_model(ModelName=name)
#         else:
#             sm.delete_endpoint_config(EndpointConfigName=name)
#         print(f"Deleted {res_type}: {name}")
#     except sm.exceptions.ClientError as e:
#         if "Could not find" in str(e):
#             print(f"{res_type} not found (skip delete): {name}")
#         else:
#             raise

# # ---------- Create Model ----------
# print("Creating SageMaker model...")
# sm.create_model(
#     ModelName=MODEL_NAME,
#     ExecutionRoleArn=ROLE_ARN,
#     PrimaryContainer={
#         "Image": IMAGE_URI,
#         "ModelDataUrl": MODEL_ARTIFACT,
#         "Environment": {
#             "HF_TASK": "text-generation"
#         }
#     }
# )
# print(f"Model created: {MODEL_NAME}")

# # ---------- Create Endpoint Config ----------
# config_name = f"{ENDPOINT_NAME}-config"
# print("Creating endpoint config...")
# sm.create_endpoint_config(
#     EndpointConfigName=config_name,
#     ProductionVariants=[{
#         "VariantName": "AllTraffic",
#         "ModelName": MODEL_NAME,
#         "InstanceType": INSTANCE_TYPE,
#         "InitialInstanceCount": 1
#     }]
# )
# print(f"Endpoint config created: {config_name}")

# # ---------- Create or Update Endpoint ----------
# existing = sm.list_endpoints(NameContains=ENDPOINT_NAME)["Endpoints"]
# if existing:
#     print("Updating existing endpoint...")
#     sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
# else:
#     print("Creating new endpoint...")
#     sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

# # ---------- Wait for Endpoint ----------
# print("Waiting for endpoint to be InService...")
# waiter = sm.get_waiter("endpoint_in_service")
# try:
#     waiter.wait(EndpointName=ENDPOINT_NAME)
#     print(f"Endpoint ready and in service: {ENDPOINT_NAME}")
# except Exception as e:
#     print(" Deployment failed:", str(e))
#     raise


import boto3
import time
import json
import os

# ---------- Load Config ----------
CONFIG_PATH = "configs/sagemaker_config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

MODEL_NAME = cfg["model_name"]
ENDPOINT_NAME = cfg["endpoint_name"]
ROLE_ARN = cfg["role_arn"]
MODEL_ARTIFACT = cfg["model_artifact"]
INSTANCE_TYPE = cfg.get("instance_type", "ml.t3.medium")
REGION = cfg.get("region", "ap-southeast-2")
IMAGE_URI = cfg.get("image_uri")

sm = boto3.client("sagemaker", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# ---------- Clean Up Old Model/Config ----------
print("Deleting existing model & config (if exist)...")
for res_type, name in [("model", MODEL_NAME), ("endpoint-config", f"{ENDPOINT_NAME}-config")]:
    try:
        if res_type == "model":
            sm.delete_model(ModelName=name)
        else:
            sm.delete_endpoint_config(EndpointConfigName=name)
        print(f"Deleted {res_type}: {name}")
    except sm.exceptions.ClientError as e:
        if "Could not find" in str(e):
            print(f" {res_type} not found: {name}")
        else:
            raise

# ---------- Create Model ----------
print("Creating new SageMaker model...")
sm.create_model(
    ModelName=MODEL_NAME,
    ExecutionRoleArn=ROLE_ARN,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": MODEL_ARTIFACT,
        "Environment": {
            "HF_TASK": "text-generation"
        }
    }
)
print(f"Model created: {MODEL_NAME}")

# ---------- Create Endpoint Config ----------
config_name = f"{ENDPOINT_NAME}-config"
print("Creating endpoint config...")
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": MODEL_NAME,
        "InstanceType": INSTANCE_TYPE,
        "InitialInstanceCount": 1
    }]
)
print(f"Endpoint config created: {config_name}")

# ---------- Create or Update Endpoint ----------
existing = sm.list_endpoints(NameContains=ENDPOINT_NAME)["Endpoints"]
if existing:
    print("Updating existing endpoint...")
    sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
else:
    print("Creating new endpoint...")
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

# ---------- Wait for Endpoint ----------
print("Waiting for endpoint to be InService...")
waiter = sm.get_waiter("endpoint_in_service")
try:
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print(f"Endpoint ready: {ENDPOINT_NAME}")
except Exception as e:
    print(" Deployment failed:", str(e))
    raise

# ---------- Test Endpoint ----------
print("Testing endpoint inference...")
test_input = json.dumps({"inputs": "Hello from MLOps!"})
try:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=test_input
    )
    result = response["Body"].read().decode("utf-8")
    print("Inference Result:", result[:300])  # preview
except Exception as e:
    print(" Endpoint test failed:", str(e))
