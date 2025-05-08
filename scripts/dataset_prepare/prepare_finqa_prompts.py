import os
import json

RAW_DIR = "data/dataset"        
OUT_DIR = "data/dataset/finqa_prompt"    

os.makedirs(OUT_DIR, exist_ok=True)

def build_prompt(entry):
    question = entry["qa"]["question"]
    answer = str(entry["qa"]["exe_ans"])  # ใช้ exe_ans เป็นคำตอบ

    # รวม context: pre_text + table + post_text
    context_parts = entry.get("pre_text", []) + flatten_table(entry.get("table", [])) + entry.get("post_text", [])
    context = " ".join(context_parts)

    prompt = f"""### Question:
{question}

### Context:
{context}

### Answer:"""

    return {
        "prompt": prompt.strip(),
        "response": answer.strip()
    }

def flatten_table(table):
    """
    แปลง table ที่เป็น list-of-rows ให้กลายเป็น list-of-sentences
    เช่น [["Year", "Revenue"], ["2020", "$1B"]] -> ["Year Revenue", "2020 $1B"]
    """
    return [" ".join(row) for row in table]

def process_split(split_name):
    input_path = os.path.join(RAW_DIR, f"{split_name}.json")
    output_path = os.path.join(OUT_DIR, f"{split_name}.jsonl")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in raw_data:
            try:
                example = build_prompt(entry)
                out_f.write(json.dumps(example) + "\n")
            except Exception as e:
                print(f"Error in {entry.get('id', 'unknown')}: {e}")

    print(f"Saved {split_name} to: {output_path} ({len(raw_data)} samples)")

if __name__ == "__main__":
    for split in ["train", "dev"]:
        process_split(split)
