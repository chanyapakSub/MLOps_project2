import os
import json
import subprocess
from datetime import datetime
import threading
import boto3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.aws_utils import read_registry, write_registry, delete_endpoint

# from scripts.aws_utils import read_registry, write_registry, delete_endpoint

# ---------- Load Config ----------
with open("configs/mlflow_config.json") as f:
    mlflow_cfg = json.load(f)

s3_bucket = mlflow_cfg["s3_bucket"]
s3_key = mlflow_cfg["s3_key"]
endpoint_name = mlflow_cfg["endpoint_name"]
registry_key = "registry/registry.json"
REGION = "ap-southeast-2"
# ---------- Train ----------
print("Training model...")
subprocess.run(["python", "scripts/trains/train.py"], check=True)

# ---------- Evaluate BEFORE Compression ----------
print("Evaluating BEFORE compression...")
subprocess.run(["python", "scripts/eval_metrics.py",
                "--model_dir", "models/finetuned_model",
                "--tag", "before_compress"], check=True)

# ---------- Compress ----------
print("Compressing model...")
subprocess.run(["python", "scripts/compress/quantize.py"], check=True)

# ---------- Evaluate AFTER Compression ----------
print("Evaluating AFTER compression...")
subprocess.run(["python", "scripts/eval_metrics.py",
                "--model_dir", "models/quantize_model",
                "--tag", "after_compress"], check=True)

# ---------- Load Metrics ----------
metrics_path = "log/metrics_eval_after_compress.json"
if not os.path.exists(metrics_path):
    raise FileNotFoundError(f"{metrics_path} not found!")

with open(metrics_path) as f:
    metrics = json.load(f)

exact_match = metrics["public"].get("exact_match", 0.0)
print("New accuracy:", exact_match)

# ---------- Upload model.tar.gz to S3 ----------
tar_path = "model.tar.gz"
if not os.path.exists(tar_path):
    raise FileNotFoundError(f"{tar_path} not found!")

s3 = boto3.client("s3" , REGION)
s3.upload_file(tar_path, s3_bucket, s3_key)
print(f"Uploaded model to: s3://{s3_bucket}/{s3_key}")

# ---------- Compare with registry ----------
registry = read_registry(s3_bucket, registry_key)
prev_acc = registry["current"]["accuracy"]
print(f"Previous accuracy: {prev_acc}")

# if exact_match < prev_acc - 0.02:
#     print("Accuracy dropped > 2% → ROLLBACK initiated.")
#     delete_endpoint(registry["current"]["endpoint"])
# else:
#     print("Accuracy OK → Deploy new model")

    # ---------- Deploy ----------
subprocess.run(["python", "mlops/sagemaker_deploy.py"], check=True)

# ---------- Schedule Endpoint Auto-Delete ----------
def schedule_endpoint_deletion(endpoint_name, delay_seconds=1800):
    def delete():
        print(f"Auto-deleting endpoint '{endpoint_name}' after 30 minutes...")
        delete_endpoint(endpoint_name)
    threading.Timer(delay_seconds, delete).start()

schedule_endpoint_deletion(endpoint_name)

    # ---------- Update registry ----------
new_version = f"v{int(registry['current']['version'][1:]) + 1}"
registry["history"].append(registry["current"])
registry["current"] = {
        "version": new_version,
        "s3_path": f"s3://{s3_bucket}/{s3_key}",
        "accuracy": exact_match,
        "timestamp": str(datetime.now()),
        "endpoint": endpoint_name
}
write_registry(s3_bucket, registry, registry_key)

print("Pipeline completed successfully.")
