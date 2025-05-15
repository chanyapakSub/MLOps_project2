#Local
# import os
# import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import Dataset
# from optimum.intel.openvino import OVConfig, OVQuantizer

# # ----- CONFIG -----
# INPUT_MODEL_DIR = "models/finetuned_model/checkpoint-1562"
# OUTPUT_DIR = "models/quantize_model_int8"
# METRICS_PATH = "log/metrics_int8.json"
# JSONL_PATH = "log/generated_answers.jsonl"
# CALIBRATION_SAMPLES = 50

# # ----- Load Tokenizer -----
# tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)

# # ----- Load Calibration Dataset from .jsonl -----
# with open(JSONL_PATH, "r", encoding="utf-8") as f:
#     records = [json.loads(line) for line in f]

# questions = [item["question"] for item in records[:CALIBRATION_SAMPLES]]
# dataset = Dataset.from_dict({"text": questions})

# def preprocess(example):
#     return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# calibration_dataset = dataset.map(preprocess, batched=True)

# # ----- Load PyTorch Model -----
# pt_model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)

# # ----- Create Quantization Config -----
# quant_config = OVConfig(quantization_type="static")

# # ----- Create Quantizer -----
# quantizer = OVQuantizer(
#     model=pt_model,
#     feature="causal-lm",
#     ov_config=quant_config,
#     task="text-generation"
# )

# # ----- Run Quantization and Save -----
# print("Starting quantization with real domain data...")
# quantizer.quantize(
#     calibration_dataset=calibration_dataset,
#     save_directory=OUTPUT_DIR
# )

# # ----- Save Tokenizer -----
# tokenizer.save_pretrained(OUTPUT_DIR)

# # ----- Measure Model Size -----
# xml_path = os.path.join(OUTPUT_DIR, "openvino_model.xml")
# if not os.path.exists(xml_path):
#     raise FileNotFoundError(f"Missing file: {xml_path}")

# size_mb = os.path.getsize(xml_path) / (1024 * 1024)
# print(f"INT8 quantized model size: {round(size_mb, 2)} MB")

# # ----- Save Metrics -----
# metrics = {
#     "model_size_mb": round(size_mb, 2),
#     "quantization": "int8",
#     "calibration_samples": CALIBRATION_SAMPLES
# }
# os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
# with open(METRICS_PATH, "w") as f:
#     json.dump(metrics, f, indent=2)

# print("Done. INT8 model saved at:", OUTPUT_DIR)




import os
import json
import tarfile
import boto3
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from optimum.intel.openvino import OVConfig, OVQuantizer

# ----- Load Config -----
with open("configs/model_config.json") as f:
    model_cfg = json.load(f)

# ----- Paths from Config -----
S3_URI = model_cfg["model_s3_path_finetuned"]
S3_BUCKET = S3_URI.split("/")[2]
S3_PREFIX = "/".join(S3_URI.split("/")[3:])
INPUT_MODEL_DIR = "models/tmp_finetuned_model"
OUTPUT_DIR = "models/quantize_model_int8"
TAR_PATH = "model.tar.gz"
S3_KEY_UPLOAD = f"{S3_PREFIX}quantize_model.tar.gz"  # เช่น Model/finetuned_model/quantize_model.tar.gz
REGION = "ap-southeast-2"
METRICS_PATH = "log/metrics_eval_after_compress.json"
JSONL_PATH = "log/generated_answers.jsonl"
CALIBRATION_SAMPLES = 50

# ----- Download finetuned model from S3 -----
def download_model_from_s3(s3_uri, local_dir):
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"✅ Found local model at {local_dir}")
        return
    print(f"☁️ Downloading model from: {s3_uri}")
    s3 = boto3.client("s3", region_name=REGION)
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            local_path = os.path.join(local_dir, os.path.relpath(s3_key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, s3_key, local_path)
    print(f"✅ Model downloaded to {local_dir}")

download_model_from_s3(S3_URI, INPUT_MODEL_DIR)

# ----- Load Tokenizer & Dataset -----
tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
questions = [item["question"] for item in records[:CALIBRATION_SAMPLES]]
dataset = Dataset.from_dict({"text": questions})

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

calibration_dataset = dataset.map(preprocess, batched=True)

# ----- Load Model & Quantize -----
pt_model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)
quant_config = OVConfig(quantization_type="static")
quantizer = OVQuantizer(model=pt_model, feature="causal-lm", ov_config=quant_config, task="text-generation")

print("🔧 Starting quantization...")
quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ----- Remove Large Weight (optional) -----
bin_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
if os.path.exists(bin_path):
    print(f"🗑️ Removing: {bin_path}")
    os.remove(bin_path)

# ----- Save Metrics -----
xml_path = os.path.join(OUTPUT_DIR, "openvino_model.xml")
size_mb = os.path.getsize(xml_path) / (1024 * 1024)
metrics = {
    "model_size_mb": round(size_mb, 2),
    "quantization": "int8",
    "calibration_samples": CALIBRATION_SAMPLES
}
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# ----- Compress Model -----
print(f"📦 Compressing {OUTPUT_DIR} → {TAR_PATH}")
with tarfile.open(TAR_PATH, "w:gz") as tar:
    tar.add(OUTPUT_DIR, arcname=".")

# ----- Upload to S3 -----
print(f"☁️ Uploading to s3://{S3_BUCKET}/{S3_KEY_UPLOAD}")
s3 = boto3.client("s3", region_name=REGION)
s3.upload_file(TAR_PATH, S3_BUCKET, S3_KEY_UPLOAD)

# ----- Verify -----
response = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY_UPLOAD)
uploaded_size = response["ContentLength"] / (1024 * 1024)
print(f"✅ Uploaded: {uploaded_size:.2f} MB")
