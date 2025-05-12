# import os
# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm

# # ---------- CONFIG ----------
# JUDGE_MODEL_DIR = "models/base_model/deepseek-7b-chat"
# GENERATED_ANSWERS_PATH = "log/generated_answers.jsonl"
# EVAL_OUTPUT_PATH = "log/llm_judge_score.json"

# # ---------- LOAD MODEL ----------
# print("Loading judge model from local path...")
# tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_DIR)
# model = AutoModelForCausalLM.from_pretrained(
#     JUDGE_MODEL_DIR,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
# model.eval()

# # ---------- LOAD GENERATED ANSWERS ----------
# if not os.path.exists(GENERATED_ANSWERS_PATH):
#     raise FileNotFoundError(f"{GENERATED_ANSWERS_PATH} not found!")

# with open(GENERATED_ANSWERS_PATH, "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# # ---------- EVALUATE ----------
# print("Scoring answers using judge model...")
# results = []
# for sample in tqdm(data):
#     question = sample["question"]
#     gt_answer = sample["ground_truth"]
#     pred_answer = sample["predicted"]

#     prompt = f"""You are a financial expert. Rate how correct and helpful this answer is to the question, from 1 to 10.

# ### Question:
# {question}

# ### Answer:
# {pred_answer}

# ### Ground Truth:
# {gt_answer}

# ### Rating (number only):"""

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=10,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     score_str = response.strip().split("Rating")[-1].strip().split()[0]
#     try:
#         score = float(score_str)
#     except:
#         score = None

#     results.append({
#         "question": question,
#         "answer": pred_answer,
#         "ground_truth": gt_answer,
#         "score": score,
#         "raw_response": response
#     })

# # ---------- SAVE RESULT ----------
# os.makedirs(os.path.dirname(EVAL_OUTPUT_PATH), exist_ok=True)
# with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)

# print(f"Judgment completed: saved to {EVAL_OUTPUT_PATH}")


import os
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

# ---------- CONFIG ----------
JUDGE_MODEL_PATH = "models/base_model/deepseek-7b-chat"
CLASSIFY_MODEL_PATH = "models/base_model/deepseek-7b-chat"
INPUT_PATH = "log/generated_answers_quantize.jsonl"
OUTPUT_PATH = "log/llm_judge_score_with_type_quantize.json"

# ---------- LOAD MODELS ----------
print("Loading judge model...")
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH, trust_remote_code=True)
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
judge_model.eval()

print("Loading classification model...")
classify_tokenizer = AutoTokenizer.from_pretrained(CLASSIFY_MODEL_PATH)
classify_model = AutoModelForCausalLM.from_pretrained(
    CLASSIFY_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
classify_model.eval()

# ---------- LOAD DATA ----------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# ---------- PROMPT TEMPLATE FOR CLASSIFICATION ----------
prompt_template = """You are a financial data expert.

Classify the following question into ONLY ONE of these types:
- quantitative
- table_aggregation
- financial_concept

Respond with ONLY ONE word: the label itself.

### Question:
{question}

### Label:"""

def extract_score(text):
    lines = text.strip().splitlines()
    for line in lines:
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", line)
        if match:
            return float(match.group(1))
    return None

# ---------- EVALUATE + CLASSIFY ----------
type_counter = Counter()
results = []
for idx, item in enumerate(tqdm(records, desc="Scoring + Classifying")):
    q = item["question"]
    a = item["predicted"]
    ref = item["ground_truth"]

    # ----- Scoring -----
    prompt = f"""You are a financial domain expert.

Rate the model's answer to the following question on a scale from 1 to 10.

Score criteria:
1. Correctness
2. Completeness
3. Clarity

Respond ONLY with the number. No explanation.

Question:
{q}

Model's Answer:
{a}

Ground Truth:
{ref}

Rating (number only):
"""
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            pad_token_id=judge_tokenizer.eos_token_id
        )
    decoded = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    score = extract_score(decoded[len(prompt):].strip())

    # ----- Classification -----
    classify_prompt = prompt_template.format(question=q)
    inputs = classify_tokenizer(classify_prompt, return_tensors="pt", truncation=True).to(classify_model.device)
    with torch.no_grad():
        outputs = classify_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
            pad_token_id=classify_tokenizer.eos_token_id
        )
    decoded_classify = classify_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    last_word = decoded_classify.strip().split()[-1]
    if last_word in {"quantitative", "table_aggregation", "financial_concept"}:
        type_label = last_word
    else:
        type_label = "unknown"

    # ----- Save result -----
    results.append({
        "question": q,
        "generated_answer": a,
        "ground_truth": ref,
        "score": score,
        "raw_response": decoded,
        "type": type_label,
        "type_raw_response": decoded_classify
    })

    type_counter[type_label] += 1
    if (idx + 1) % 5 == 0:
        print(f"\n Processed {idx+1}/{len(records)}")
        print("Current type distribution:", dict(type_counter))

# ---------- SAVE ----------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n Saved full result with score and type to: {OUTPUT_PATH}")
print("Final type distribution:", dict(type_counter))
