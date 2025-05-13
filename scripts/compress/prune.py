import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from optimum.intel.pune import PUNeConfig, PUNeSparsifier

# ----- CONFIG -----
INPUT_MODEL_DIR = "models/finetuned_model/checkpoint-1562"
OUTPUT_DIR = "models/pune_model"
METRICS_PATH = "log/metrics_pune.json"
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
model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)

# ----- Create PUNe Config -----
pune_config = PUNeConfig(
    approach="pst_nm",          # post-training structured sparsity
    sparsity_level=0.5,         # 50% sparsity
    block_size=(1, 4),          # block pattern for 2:4
    pattern="2:4"               # classic Nvidia-style pattern
)

# ----- Create PUNe Sparsifier -----
sparsifier = PUNeSparsifier(
    model=model,
    config=pune_config,
    task_type="text-generation",
    tokenizer=tokenizer
)

# ----- Run Sparsification and Save -----
print("Starting PUNe sparsification with real domain data...")
sparsifier.sparsify(
    calibration_dataset=calibration_dataset,
    save_directory=OUTPUT_DIR
)

# ----- Save Tokenizer -----
tokenizer.save_pretrained(OUTPUT_DIR)

# ----- Save Sparsity Metrics -----
metrics = {
    "compression": "pune",
    "pattern": "2:4",
    "sparsity_level": 0.5,
    "calibration_samples": CALIBRATION_SAMPLES
}
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("Done. PUNe sparsified model saved at:", OUTPUT_DIR)
