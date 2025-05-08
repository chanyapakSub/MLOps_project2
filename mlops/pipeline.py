import os
import json
import mlflow
import subprocess
from datetime import datetime

# ---------- Load Config ----------
with open("configs/mlflow_config.json") as f:
    mlflow_cfg = json.load(f)

mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
mlflow.set_experiment(mlflow_cfg["experiment_name"])

with mlflow.start_run():
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")

    # ---------- Training ----------
    print("Training model...")
    subprocess.run(["python", "scripts/trains/train.py"], check=True)

    # ---------- Compression ----------
    print("Compressing model...")
    subprocess.run(["python", "scripts/compress/quantize.py"], check=True)

    # ---------- Load Metrics ----------
    metrics_path = "log/metrics.json"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"{metrics_path} not found!")

    with open(metrics_path) as f:
        metrics = json.load(f)

    print(" Metrics:", metrics)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(metrics_path, artifact_path="log")

    # ---------- Log Model ----------
    model_path = "models/hybrid_model/model.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found!")

    mlflow.log_artifact(model_path, artifact_path="compressed_model")

    # ---------- Conditional Register ----------
    should_register = True
    if mlflow_cfg.get("upload_only_if_metric_above"):
        for metric, threshold in mlflow_cfg["upload_only_if_metric_above"].items():
            val = metrics.get(metric, 0)
            if val < threshold:
                should_register = False
                print(f" Skipping register: {metric}={val} < {threshold}")
                break

    if mlflow_cfg.get("register_model") and should_register:
        print(" Registering model...")
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/compressed_model",
            name=mlflow_cfg["registry_model_name"]
        )

    print(" Pipeline complete.")

