# import os
# import json
# from transformers import AutoTokenizer
# from optimum.intel.openvino import OVModelForCausalLM

# # ----- CONFIG -----
# INPUT_MODEL_DIR = "models/finetuned_model/checkpoint-1562"
# OUTPUT_DIR = "models/quantize_model"
# METRICS_PATH = "log/metrics.json"

# # ----- Load and Export Model -----
# model = OVModelForCausalLM.from_pretrained(INPUT_MODEL_DIR, export=True, use_cache=True)
# tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# # ----- Calculate Model Size -----
# xml_path = os.path.join(OUTPUT_DIR, "openvino_model.xml")
# if not os.path.exists(xml_path):
#     raise FileNotFoundError(f"Missing file: {xml_path}")

# size_mb = os.path.getsize(xml_path) / (1024 * 1024)
# print(f"📦 Quantized model size: {round(size_mb, 2)} MB")

# # ----- Save Metrics -----
# metrics = {
#     "model_size_mb": round(size_mb, 2)
# }
# os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
# with open(METRICS_PATH, "w") as f:
#     json.dump(metrics, f, indent=2)

# print("✅ Done. Model ready at:", OUTPUT_DIR)

import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from optimum.intel.openvino import OVConfig, OVQuantizer

# ----- CONFIG -----
INPUT_MODEL_DIR = "models/finetuned_model/checkpoint-1562"
OUTPUT_DIR = "models/quantize_model_int8"
METRICS_PATH = "log/metrics_int8.json"
JSONL_PATH = "log/generated_answers.jsonl"
CALIBRATION_SAMPLES = 50

# ----- Load Tokenizer -----
tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)

# ----- Load Calibration Dataset from .jsonl -----
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

questions = [item["question"] for item in records[:CALIBRATION_SAMPLES]]
dataset = Dataset.from_dict({"text": questions})

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

calibration_dataset = dataset.map(preprocess, batched=True)

# ----- Load PyTorch Model -----
pt_model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)

# ----- Create Quantization Config -----
quant_config = OVConfig(quantization_type="static")

# ----- Create Quantizer -----
quantizer = OVQuantizer(
    model=pt_model,
    feature="causal-lm",
    ov_config=quant_config,
    task="text-generation"
)

# ----- Run Quantization and Save -----
print("Starting quantization with real domain data...")
quantizer.quantize(
    calibration_dataset=calibration_dataset,
    save_directory=OUTPUT_DIR
)

# ----- Save Tokenizer -----
tokenizer.save_pretrained(OUTPUT_DIR)

# ----- Measure Model Size -----
xml_path = os.path.join(OUTPUT_DIR, "openvino_model.xml")
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"Missing file: {xml_path}")

size_mb = os.path.getsize(xml_path) / (1024 * 1024)
print(f"INT8 quantized model size: {round(size_mb, 2)} MB")

# ----- Save Metrics -----
metrics = {
    "model_size_mb": round(size_mb, 2),
    "quantization": "int8",
    "calibration_samples": CALIBRATION_SAMPLES
}
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("Done. INT8 model saved at:", OUTPUT_DIR)
