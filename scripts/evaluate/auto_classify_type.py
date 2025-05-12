import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import Counter

# ----- CONFIG -----
MODEL_DIR = "models/base_model/deepseek-7b-chat"
INPUT_PATH = "log/llm_judge_score_with_type.json"
OUTPUT_PATH = "log/llm_judge_score_with_type_reclassified.json"

# ----- PROMPT TEMPLATE -----
prompt_template = """You are a financial data expert.

Classify the following question into ONLY ONE of these types:
- quantitative
- table_aggregation
- financial_concept

Respond with ONLY ONE word: the label itself.

### Question:
{question}

### Label:"""

# ----- LOAD MODEL -----
print("🔍 Loading LLM for classification...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ----- LOAD DATA -----
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ----- CLASSIFY -----
type_counter = Counter()
for idx, item in enumerate(tqdm(data, desc="🔎 Reclassifying type")):
    question = item.get("question", "")
    answer = item.get("answer") or item.get("generated_answer") or item.get("predicted", "")
    prompt = prompt_template.format(question=question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    last_word = decoded.strip().split()[-1]

    if last_word in {"quantitative", "table_aggregation", "financial_concept"}:
        new_type = last_word
    else:
        new_type = "unknown"

    item["type"] = new_type
    item["type_raw_response"] = decoded
    type_counter[new_type] += 1

    if (idx + 1) % 5 == 0:
        print(f"\n📊 Processed {idx+1}/{len(data)} questions")
        print("Current type distribution:", dict(type_counter))

# ----- SAVE RESULT -----
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved updated file with reclassified types to: {OUTPUT_PATH}")
print("📈 Final type distribution:", dict(type_counter))
