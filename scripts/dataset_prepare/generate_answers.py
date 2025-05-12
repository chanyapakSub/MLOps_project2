import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM 
import torch

data_path = "data/dataset/test.json"  # test set
model_dir = "models/quantize_model"
output_file = "log/generated_answers_quantize.jsonl"

# ----- Load test set -----
with open(data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# ----- Load model -----
print("Loading model from", model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
model = OVModelForCausalLM.from_pretrained(model_dir)
model.eval()

def generate_answer(input_text):
    prompt = f"{input_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()

# ----- Generate and save answers -----
os.makedirs("log", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as fout:
    for item in tqdm(test_data):
        question = item["qa"]["question"]
        gt_answer = item["qa"]["answer"]
        pred_answer = generate_answer(question)

        fout.write(json.dumps({
            "question": question,
            "ground_truth": gt_answer,
            "predicted": pred_answer
        }, ensure_ascii=False) + "\n")

print(f"Generated answers saved to {output_file}")