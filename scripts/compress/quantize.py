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



# === quantize.py === 
import os
import json
import tarfile
import boto3
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel.openvino import OVConfig, OVQuantizer

# ----- Load Config -----
with open("configs/model_config.json") as f:
    model_cfg = json.load(f)

region = "ap-southeast-2"
s3_finetuned_path = model_cfg["model_s3_path_finetuned"] + "model.tar.gz"
s3_compressed_prefix = "Model/compressed_model/"
output_dir = "models/quantize_model"
metrics_path = "log/metrics_eval_after_compress.json"
jsonl_path = "log/generated_answers.jsonl"
calibration_samples = 50

# ----- DOWNLOAD FROM S3 -----
def download_and_extract_model(s3_uri, extract_to):
    s3 = boto3.client("s3", region_name=region)
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    tar_path = "temp_model.tar.gz"
    s3.download_file(bucket, key, tar_path)
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    os.remove(tar_path)
    print(f"✅ Downloaded and extracted model from {s3_uri}")

# ----- UPLOAD TO S3 -----
def upload_dir_to_s3(local_dir, s3_uri_prefix):
    s3 = boto3.client("s3", region_name=region)
    bucket, prefix = s3_uri_prefix.replace("s3://", "").split("/", 1)
    for root, _, files in os.walk(local_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, local_dir)
            s3_path = os.path.join(prefix, relative_path).replace("\\", "/")
            s3.upload_file(full_path, bucket, s3_path)
            print(f"📤 Uploaded {relative_path} to s3://{bucket}/{s3_path}")

# ----- Load Tokenizer & Model -----
download_and_extract_model(s3_finetuned_path, "tmp_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("tmp_finetuned_model")
pt_model = AutoModelForCausalLM.from_pretrained("tmp_finetuned_model")

# ----- Load Calibration Dataset -----
with open(jsonl_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
questions = [item["question"] for item in records[:calibration_samples]]
dataset = Dataset.from_dict({"text": questions})

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

calibration_dataset = dataset.map(preprocess, batched=True)

# ----- Quantization -----
quant_config = OVConfig(quantization_type="static")
quantizer = OVQuantizer(
    model=pt_model,
    feature="causal-lm",
    ov_config=quant_config,
    task="text-generation"
)

print("🔧 Running quantization...")
quantizer.quantize(
    calibration_dataset=calibration_dataset,
    save_directory=output_dir
)

# ----- Save Tokenizer -----
tokenizer.save_pretrained(output_dir)

# ----- Measure Size & Save -----
xml_path = os.path.join(output_dir, "openvino_model.xml")
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"Missing file: {xml_path}")

size_mb = os.path.getsize(xml_path) / (1024 * 1024)
metrics = {
    "model_size_mb": round(size_mb, 2),
    "quantization": "int8",
    "calibration_samples": calibration_samples
}
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# ----- Upload to S3 -----
print("📤 Uploading quantized model to S3...")
upload_dir_to_s3(output_dir, f"s3://{model_cfg['s3_bucket']}/{s3_compressed_prefix}")
print(f"✅ Quantized model uploaded to: s3://{model_cfg['s3_bucket']}/{s3_compressed_prefix} ({round(size_mb, 2)} MB)")
