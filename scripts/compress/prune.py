import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from neural_compressor.config import WeightPruningConfig
from neural_compressor.training import prepare_compression
from datasets import Dataset
from torch.utils.data import DataLoader

# ----- CONFIG -----
INPUT_MODEL_DIR = "models/finetuned_model/checkpoint-1562"
OUTPUT_DIR = "models/pruned_model"
METRICS_PATH = "log/metrics_prune.json"
JSONL_PATH = "log/generated_answers.jsonl"
CALIBRATION_SAMPLES = 50
BATCH_SIZE = 4

# ----- Load Model -----
model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)

# ----- Load Calibration Data -----
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

questions = [item["question"] for item in records[:CALIBRATION_SAMPLES]]
dataset = Dataset.from_dict({"text": questions})

def preprocess(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(preprocess, batched=True)

# ✅ ใช้ collate_fn เพื่อให้ batch เป็น tensor
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=default_data_collator
)

# ----- Pruning Config -----
TARGET_SPARSITY = 0.5
PRUNE_PATTERN = "2:4"

config = WeightPruningConfig(
    pruning_type="magnitude",
    target_sparsity=TARGET_SPARSITY,
    pattern=PRUNE_PATTERN,
    pruning_op_types=["Linear"],
    start_step=0,
    end_step=1
)


# ----- Prepare for Pruning -----
compression_manager = prepare_compression(model, config)

# ----- Run Pruning Step -----
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for batch in dataloader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        model(**inputs)

# ----- Save Pruned Model -----
compression_manager.save(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ----- Save Metrics -----
metrics = {
    "compression": "structured_prune",
    "target_sparsity": TARGET_SPARSITY,
    "pattern": PRUNE_PATTERN,
    "calibration_samples": CALIBRATION_SAMPLES
}

os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Pruned model saved at: {OUTPUT_DIR}")
