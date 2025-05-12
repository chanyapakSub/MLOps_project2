import os 
import json
import mlflow
import subprocess
from datetime import datetime
import boto3

# ---------- Load Config ----------
with open("configs/mlflow_config.json") as f:
    mlflow_cfg = json.load(f)

mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
mlflow.set_experiment(mlflow_cfg["experiment_name"])

with mlflow.start_run():
    run_id = mlflow.active_run().info.run_id
    print(f" Run ID: {run_id}")

    # ---------- Training ----------
    print("Training model...")
    subprocess.run(["python", "scripts/trains/train.py"], check=True)

    # ---------- Evaluate BEFORE Compression ----------
    print("Evaluating BEFORE compression...")
    subprocess.run(["python", "scripts/eval_metrics.py",
                    "--model_dir", "models/finetuned_model",
                    "--tag", "before_compress"], check=True)

    # ---------- Compression ----------
    print("Compressing model...")
    subprocess.run(["python", "scripts/compress/quantize.py"], check=True)

    # ---------- Evaluate AFTER Compression ----------
    print("Evaluating AFTER compression...")
    subprocess.run(["python", "scripts/eval_metrics.py",
                    "--model_dir", "models/quantize_model",
                    "--tag", "after_compress"], check=True)

    # ---------- Load Post-Compression Metrics ----------
    metrics_path = "log/metrics.json"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"{metrics_path} not found!")

    with open(metrics_path) as f:
        metrics = json.load(f)

    print("Metrics (after compress):", metrics)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(metrics_path, artifact_path="eval")

    # ---------- Log Model Artifact ----------
    model_path = "models/quantize_model/model.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found!")

    mlflow.log_artifact(model_path, artifact_path="model")

    # ---------- Log Model Size ----------
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    mlflow.log_metric("model_size_mb", round(model_size_mb, 2))

    # ---------- Upload to S3 ----------
    print("Uploading model.tar.gz to S3...")
    tar_path = "model.tar.gz"
    s3_bucket = mlflow_cfg["s3_bucket"]
    s3_key = mlflow_cfg["s3_key"]
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"{tar_path} not found!")

    s3 = boto3.client("s3")
    s3.upload_file(tar_path, s3_bucket, s3_key)
    print(f"Uploaded to s3://{s3_bucket}/{s3_key}")

    # ---------- Conditional Register ----------
    should_register = True
    if mlflow_cfg.get("upload_only_if_metric_above"):
        for metric, threshold in mlflow_cfg["upload_only_if_metric_above"].items():
            val = metrics.get(metric, 0)
            if val < threshold:
                should_register = False
                print(f"Skipping register: {metric}={val} < {threshold}")
                break

    if mlflow_cfg.get("register_model") and should_register:
        print("Registering model to MLflow...")
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=mlflow_cfg["registry_model_name"]
        )

    print("Pipeline completed.")
