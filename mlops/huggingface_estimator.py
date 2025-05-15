import sagemaker
from sagemaker.huggingface import HuggingFace 
import json

# ===== LOAD CONFIGS =====
with open("configs/train_config.json") as f:
    train_cfg = json.load(f)

with open("configs/model_config.json") as f:
    model_cfg = json.load(f)

with open("configs/sagemaker_config.json") as f:
    sm_cfg = json.load(f)

# ===== Set Up HuggingFace Estimator =====
huggingface_estimator = HuggingFace(
    entry_point="train.py",  # สคริปต์ฝึกโมเดลที่รองรับ argparse แล้ว
    source_dir=".",  # ตำแหน่งไฟล์ train.py และ configs/
    instance_type=sm_cfg["instance_type"],  # ex: ml.t3.medium
    instance_count=1,
    role=sm_cfg["role_arn"],
    transformers_version="4.17",
    pytorch_version="1.10",
    py_version="py38", 
    region=sm_cfg["region"],

    # ===== Hyperparameters to pass into train.py via argparse
    hyperparameters={
        "model_s3_path": model_cfg["model_s3_path"],
        "train_file": model_cfg["train_file"],
        "eval_file": model_cfg["eval_file"],
        "fp16": train_cfg["fp16"]
    }
)

# ===== Run Training Job on SageMaker =====
huggingface_estimator.fit({
    "train": model_cfg["train_file"],
    "eval": model_cfg["eval_file"]
})
