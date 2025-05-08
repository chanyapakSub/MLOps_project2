import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel import INCQuantizer, QuantizationConfig

# ----- Path Config -----
INPUT_MODEL_DIR = "models/finetuned_model"
OUTPUT_DIR = "models/quantize_model"
METRICS_PATH = "log/metrics.json"

# ----- Load model -----
print("Loading model from:", INPUT_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(INPUT_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_DIR)

# ----- Quantization -----
print("Performing dynamic quantization...")
quantizer = INCQuantizer.from_pretrained(model)
q_config = QuantizationConfig(approach="dynamic")

os.makedirs(OUTPUT_DIR, exist_ok=True)
quantizer.quantize(
    save_directory=OUTPUT_DIR,
    quantization_config=q_config
)

# ----- Save tokenizer (optional) -----
tokenizer.save_pretrained(OUTPUT_DIR)

# ----- Evaluate (mock metrics for now) -----
onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")
size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

# ----- Save metrics -----
os.makedirs("log", exist_ok=True)
metrics = {
    "rougeL": 0.31,
    "bleu": 0.42,
    "size_mb": round(size_mb, 2)
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)

print("Quantized model saved to:", onnx_path)
print("Metrics saved to:", METRICS_PATH)
