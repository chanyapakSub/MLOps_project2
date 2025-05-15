# import os
# import json
# import subprocess
# from datetime import datetime
# import threading
# import boto3
# import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from scripts.aws_utils import read_registry, write_registry, delete_endpoint

# # ---------- Load Config ----------
# with open("configs/sagemaker_config.json") as f:
#     sm_cfg = json.load(f)

# s3_bucket = sm_cfg["s3_bucket"]
# s3_key = sm_cfg["s3_key"]
# endpoint_name = sm_cfg["endpoint_name"]
# registry_key = "registry/registry.json"
# REGION = sm_cfg["region"]

# # ---------- Train on SageMaker ----------
# print("🚀 Launching SageMaker training job...")
# subprocess.run(["python", "mlops/huggingface_estimator.py"], check=True)

# # ---------- Evaluate BEFORE Compression ----------
# print("📊 Evaluating BEFORE compression...")
# subprocess.run(["python", "scripts/eval_metrics.py",
#                 "--model_dir", "models/finetuned_model",
#                 "--tag", "before_compress"], check=True)

# # ---------- Compress ----------
# print("🔧 Compressing model...")
# subprocess.run(["python", "scripts/compress/quantize.py"], check=True)

# # ---------- Evaluate AFTER Compression ----------
# print("📊 Evaluating AFTER compression...")
# subprocess.run(["python", "scripts/eval_metrics.py",
#                 "--model_dir", "models/quantize_model",
#                 "--tag", "after_compress"], check=True)

# # ---------- Load Metrics ----------
# metrics_path = "log/metrics_eval_after_compress.json"
# if not os.path.exists(metrics_path):
#     raise FileNotFoundError(f"{metrics_path} not found!")

# with open(metrics_path) as f:
#     metrics = json.load(f)

# exact_match = metrics["public"].get("exact_match", 0.0)
# print(f"📈 New accuracy: {exact_match:.4f}")

# # ---------- Upload model.tar.gz to S3 ----------
# tar_path = "model.tar.gz"
# if not os.path.exists(tar_path):
#     raise FileNotFoundError(f"{tar_path} not found!")

# s3 = boto3.client("s3", region_name=REGION)
# s3.upload_file(tar_path, s3_bucket, s3_key)
# print(f"✅ Uploaded model to: s3://{s3_bucket}/{s3_key}")

# # ---------- Compare with registry ----------
# registry = read_registry(s3_bucket, registry_key)
# prev_acc = registry["current"]["accuracy"]
# print(f"📉 Previous accuracy: {prev_acc:.4f}")

# # if exact_match < prev_acc - 0.02:
# #     print("⚠️ Accuracy dropped > 2% → ROLLBACK initiated.")
# #     delete_endpoint(registry["current"]["endpoint"])
# # else:
# #     print("✅ Accuracy OK → Deploying new model...")

# # ---------- Deploy new model ----------
# print("🚀 Deploying to SageMaker Endpoint...")
# subprocess.run(["python", "mlops/sagemaker_deploy.py"], check=True)

# # ---------- Schedule Endpoint Auto-Delete ----------
# def schedule_endpoint_deletion(endpoint_name, delay_seconds=1800):
#     def delete():
#         print(f"🧹 Auto-deleting endpoint '{endpoint_name}' after 30 minutes...")
#         delete_endpoint(endpoint_name)
#     threading.Timer(delay_seconds, delete).start()

# schedule_endpoint_deletion(endpoint_name)

# # ---------- Update registry ----------
# new_version = f"v{int(registry['current']['version'][1:]) + 1}"
# registry["history"].append(registry["current"])
# registry["current"] = {
#     "version": new_version,
#     "s3_path": f"s3://{s3_bucket}/{s3_key}",
#     "accuracy": exact_match,
#     "timestamp": str(datetime.now()),
#     "endpoint": endpoint_name
# }
# write_registry(s3_bucket, registry, registry_key)

# print("🎯 Pipeline completed successfully.")


# === pipeline.py === 
import os
import json
import subprocess
import threading
import boto3
import shutil
from datetime import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.aws_utils import read_registry, write_registry, delete_endpoint

# ---------- Load Config ----------
with open("configs/model_config.json") as f:
    model_cfg = json.load(f)

with open("configs/sagemaker_config.json") as f:
    sm_cfg = json.load(f)

s3_bucket = sm_cfg["s3_bucket"]
s3_key_finetuned = model_cfg["model_s3_path_finetuned"].replace(f"s3://{s3_bucket}/", "").rstrip("/") + "/model.tar.gz"
s3_key_compressed = sm_cfg["s3_key"]
endpoint_name = sm_cfg["endpoint_name"]
registry_key = "registry/registry.json"
region = sm_cfg["region"]

# ---------- Train Locally ----------
print("\U0001F680 Training model locally...")
subprocess.run([
    "python", "scripts/trains/train.py",
    "--model_s3_path", model_cfg["model_s3_path"],
    "--train_file", model_cfg["train_file"],
    "--eval_file", model_cfg["eval_file"],
    "--output_dir", model_cfg["output_dir"],
    "--upload_s3_path", f"s3://{s3_bucket}/{s3_key_finetuned}"
], check=True)

# ---------- Evaluate BEFORE Compression ----------
print("\U0001F4CA Evaluating BEFORE compression...")
subprocess.run(["python", "scripts/eval_metrics.py",
                "--model_dir", model_cfg["output_dir"],
                "--tag", "before_compress"], check=True)

# ---------- Compress ----------
print("\U0001F527 Compressing model...")
subprocess.run(["python", "quantize.py"], check=True)

# ---------- Evaluate AFTER Compression ----------
print("\U0001F4CA Evaluating AFTER compression...")
subprocess.run(["python", "scripts/eval_metrics.py",
                "--model_dir", "models/quantize_model",
                "--tag", "after_compress"], check=True)

# ---------- Load Metrics ----------
metrics_path = "log/metrics_eval_after_compress.json"
with open(metrics_path) as f:
    metrics = json.load(f)

exact_match = metrics["public"].get("exact_match", 0.0)
print(f"\U0001F4C8 New accuracy: {exact_match:.4f}")

# ---------- Upload compressed model ----------
tar_path = "model.tar.gz"
s3 = boto3.client("s3", region_name=region)

# delete old fine-tuned model
try:
    s3.delete_object(Bucket=s3_bucket, Key=s3_key_finetuned)
    print("🗑️ Deleted previous fine-tuned model from S3")
except:
    pass

# upload new compressed model
s3.upload_file(tar_path, s3_bucket, s3_key_compressed)
print(f"Uploaded compressed model to: s3://{s3_bucket}/{s3_key_compressed}")

# ---------- Deploy new model ----------
print("\U0001F680 Deploying to SageMaker Endpoint...")
subprocess.run(["python", "mlops/sagemaker_deploy.py"], check=True)

# ---------- Schedule Auto Delete Endpoint ----------
def schedule_endpoint_deletion(endpoint_name, delay_seconds=1800):
    def delete():
        print(f" Auto-deleting endpoint '{endpoint_name}' after 30 minutes...")
        delete_endpoint(endpoint_name)
    threading.Timer(delay_seconds, delete).start()

schedule_endpoint_deletion(endpoint_name)

# ---------- Update registry ----------
registry = read_registry(s3_bucket, registry_key)
new_version = f"v{int(registry['current']['version'][1:]) + 1}"
registry["history"].append(registry["current"])
registry["current"] = {
    "version": new_version,
    "s3_path": f"s3://{s3_bucket}/{s3_key_compressed}",
    "accuracy": exact_match,
    "timestamp": str(datetime.now()),
    "endpoint": endpoint_name
}
write_registry(s3_bucket, registry, registry_key)

# ---------- Cleanup local ----------
shutil.rmtree(model_cfg["output_dir"], ignore_errors=True)
os.remove("model.tar.gz")
print("Pipeline complete. Cleaned up local model.")

