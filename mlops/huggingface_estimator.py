import json
from sagemaker.huggingface import HuggingFace
import sagemaker

# ===== LOAD CONFIG =====
with open("configs/sagemaker_config.json") as f:
    sm_cfg = json.load(f)

with open("configs/model_config.json") as f:
    model_cfg = json.load(f)

# ===== HuggingFace Estimator =====
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir=".",
    instance_type="ml.t3.medium",
    instance_count=1,
    role=sm_cfg["role_arn"],
    image_uri="763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:1.10-transformers4.17-cpu-py38-ubuntu20.04",
    hyperparameters={
        "model_s3_path": model_cfg["model_s3_path"],
        "train_file": model_cfg["train_file"],
        "eval_file": model_cfg["eval_file"],
        "output_dir": model_cfg["output_dir"],
        "fp16": False
    }
)

# ===== Run SageMaker Training Job =====
huggingface_estimator.fit({
    "train": model_cfg["train_file"],
    "eval": model_cfg["eval_file"]
})
