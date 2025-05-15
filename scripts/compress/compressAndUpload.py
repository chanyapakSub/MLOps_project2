# import os
# import json
# import tarfile
# import boto3
# import mlflow

# # --- CONFIG ---
# MODEL_DIR = "models/quantize_model"
# TAR_PATH = "model.tar.gz"
# S3_BUCKET = "modelmlopss"
# S3_KEY = "Model/model.tar.gz"
# # CONFIG_PATH = "configs/mlflow_config.json"
# region_name=REGION = "ap-southeast-2"

# # --- Load MLflow config ---
# # with open(CONFIG_PATH, "r") as f:
# #     mlflow_cfg = json.load(f)

# # mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
# # mlflow.set_experiment(mlflow_cfg["experiment_name"])

# def compress_model():
#     print(f"📦 Compressing: {MODEL_DIR}")
#     with tarfile.open(TAR_PATH, "w:gz") as tar:
#         tar.add(MODEL_DIR, arcname=".")
#     print(f"✅ Compressed to: {TAR_PATH}")

# def upload_to_s3():
#     print(f"☁️ Uploading to s3://{S3_BUCKET}/{S3_KEY}")
#     s3 = boto3.client("s3" , region_name=REGION)
#     s3.upload_file(TAR_PATH, S3_BUCKET, S3_KEY)
#     print("✅ Upload complete.")

# def verify_upload():
#     s3 = boto3.client("s3" ,region_name=REGION)
#     response = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
#     size_mb = response["ContentLength"] / (1024 * 1024)
#     print(f"📁 Verified on S3 ({size_mb:.2f} MB)")
#     return round(size_mb, 2)

# # --- MAIN ---
# if __name__ == "__main__":
#     if not os.path.exists(MODEL_DIR):
#         raise FileNotFoundError(f"{MODEL_DIR} not found!")

#     compress_model()
#     upload_to_s3()
#     size_mb = verify_upload()

#     # # Log to MLflow
#     # with mlflow.start_run() as run:
#     #     run_id = run.info.run_id
#     #     print(f"🔗 MLflow Run ID: {run_id}")

#     #     mlflow.log_param("model_dir", MODEL_DIR)
#     #     mlflow.log_param("s3_path", f"s3://{S3_BUCKET}/{S3_KEY}")
#     #     mlflow.log_metric("model_size_mb", size_mb)
#     #     mlflow.log_artifact(TAR_PATH, artifact_path="compressed")

#     #     # Optional: register to MLflow Registry
#     #     if mlflow_cfg.get("register_model", False):
#     #         print("📚 Registering model to MLflow Registry...")
#     #         mlflow.register_model(
#     #             model_uri=f"runs:/{run_id}/compressed",
#     #             name=mlflow_cfg["registry_model_name"]
#     #         )

#         # print("✅ Logging complete.")



import os 
import json
import tarfile
import boto3

# --- CONFIG ---
MODEL_DIR = "models/quantize_model"
TAR_PATH = "model.tar.gz"
S3_BUCKET = "modelmlopss"
S3_KEY = "Model/model.tar.gz"
REGION = "ap-southeast-2"

def compress_model():
    print(f"📦 Compressing: {MODEL_DIR}")
    with tarfile.open(TAR_PATH, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname=".")
    print(f"✅ Compressed to: {TAR_PATH}")

def upload_to_s3():
    print(f"☁️ Uploading to s3://{S3_BUCKET}/{S3_KEY}")
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(TAR_PATH, S3_BUCKET, S3_KEY)
    print("✅ Upload complete.")

def verify_upload():
    s3 = boto3.client("s3", region_name=REGION)
    response = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
    size_mb = response["ContentLength"] / (1024 * 1024)
    print(f"📁 Verified on S3 ({size_mb:.2f} MB)")
    return round(size_mb, 2)

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"{MODEL_DIR} not found!")

    compress_model()
    upload_to_s3()
    verify_upload()
