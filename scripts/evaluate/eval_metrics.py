# import os
# import json
# import time
# import argparse
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from evaluate import load

# # ---------- Parse Arguments ----------
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_dir", required=True, help="Path to model directory")
# parser.add_argument("--tag", default="latest", help="Tag to append to output file")
# parser.add_argument("--test_path", default="data/dataset/test.json", help="Public test set")
# parser.add_argument("--private_test_path", default="data/dataset/private_test.json", help="Private test set")
# args = parser.parse_args()

# # ---------- Load Model ----------
# print(f" Loading model from {args.model_dir}")
# model = AutoModelForCausalLM.from_pretrained(args.model_dir)
# tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

# rouge = load("rouge")
# bleu = load("bleu")

# # ---------- Load Dataset ----------
# def load_data(path):
#     with open(path) as f:
#         return json.load(f)

# test_data = load_data(args.test_path)
# private_data = load_data(args.private_test_path)

# # ---------- Evaluation Function ----------
# def evaluate_dataset(dataset, has_answer=True):
#     predictions, references = [], []
#     em = 0
#     total_latency = 0
#     total_tokens = 0

#     for item in tqdm(dataset, desc="Evaluating"):
#         question = item["qa"]["question"]
#         expected = item["qa"]["answer"] if has_answer else None

#         inputs = tokenizer(question, return_tensors="pt")
#         start = time.time()
#         outputs = model.generate(**inputs, max_new_tokens=64)
#         latency = time.time() - start
#         total_latency += latency

#         decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         predictions.append(decoded)
#         if has_answer:
#             references.append(expected)
#             if decoded.strip().lower() == expected.strip().lower():
#                 em += 1

#         total_tokens += outputs[0].shape[0]

#     metrics = {}
#     if has_answer:
#         metrics["rougeL"] = rouge.compute(predictions=predictions, references=references)["rougeL"]
#         metrics["bleu"] = bleu.compute(predictions=predictions, references=references)["bleu"]
#         metrics["exact_match"] = round(em / len(dataset), 4)
#     metrics["latency_sec"] = round(total_latency / len(dataset), 4)
#     metrics["throughput_tokens_per_sec"] = round(total_tokens / total_latency, 2)
#     return metrics

# # ---------- Run Evaluation ----------
# public_metrics = evaluate_dataset(test_data, has_answer=True)
# private_metrics = evaluate_dataset(private_data, has_answer=False)

# # ---------- Save ----------
# os.makedirs("log", exist_ok=True)
# output_path = f"log/metrics_eval_{args.tag}.json"
# with open(output_path, "w") as f:
#     json.dump({
#         "model_dir": args.model_dir,
#         "tag": args.tag,
#         "public": public_metrics,
#         "private": private_metrics
#     }, f, indent=2)

# print(f" Evaluation complete. Metrics saved to: {output_path}")


import os
import json
import time
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from evaluate import load
from optimum.intel.openvino import OVModelForCausalLM  # ใช้ OpenVINO model loader

# ---------- Parse Arguments or Load from Config ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="Path to model directory")
parser.add_argument("--tag", default="latest", help="Tag to append to output file")
parser.add_argument("--test_path", help="Public test set")
parser.add_argument("--private_test_path", help="Private test set")
args, unknown = parser.parse_known_args()

if not args.model_dir:
    with open("configs/model_config.json") as f:
        cfg = json.load(f)
    args.model_dir = "models/quantize_model"
    args.tag = "after_compress"
    args.test_path = "data/dataset/test.json"
    args.private_test_path = "data/dataset/private_test.json"

# ---------- Load Model ----------
print(f"📦 Loading OpenVINO model from {args.model_dir}")
model = OVModelForCausalLM.from_pretrained(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

# ---------- Compute Model Stats ----------
def get_model_disk_size(model_dir):
    total_bytes = 0
    for fname in os.listdir(model_dir):
        if fname.endswith((".bin", ".xml", ".safetensors", ".pt")):
            total_bytes += os.path.getsize(os.path.join(model_dir, fname))
    return round(total_bytes / (1024 * 1024), 2)  # MB

model_disk_size_mb = get_model_disk_size(args.model_dir)
print(f"🧠 Model file size: {model_disk_size_mb} MB")

# ---------- Load Metrics ----------
rouge = load("rouge")
bleu = load("bleu")
bertscore = load("bertscore")

# ---------- Load Dataset ----------
def load_data(path):
    with open(path) as f:
        return json.load(f)

test_data = load_data(args.test_path)
private_data = load_data(args.private_test_path)

# ---------- Evaluation Function ----------
def evaluate_dataset(dataset, has_answer=True):
    predictions, references = [], []
    em = 0
    total_latency = 0
    total_tokens = 0

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["qa"]["question"]
        expected = item["qa"]["answer"] if has_answer else None

        inputs = tokenizer(question, return_tensors="pt")
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=64)
        latency = time.time() - start
        total_latency += latency

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded)
        if has_answer:
            references.append(expected)
            if decoded.strip().lower() == expected.strip().lower():
                em += 1

        total_tokens += outputs[0].shape[0]

    metrics = {}
    if has_answer:
        metrics["rougeL"] = rouge.compute(predictions=predictions, references=references)["rougeL"]
        metrics["bleu"] = bleu.compute(predictions=predictions, references=references)["bleu"]
        metrics["exact_match"] = round(em / len(dataset), 4)
        metrics["bertscore_f1"] = sum(
            bertscore.compute(predictions=predictions, references=references, lang="en")["f1"]
        ) / len(predictions)

    metrics["latency_sec"] = round(total_latency / len(dataset), 4)
    metrics["throughput_tokens_per_sec"] = round(total_tokens / total_latency, 2)
    return metrics

# ---------- Run Evaluation ----------
public_metrics = evaluate_dataset(test_data, has_answer=True)
private_metrics = evaluate_dataset(private_data, has_answer=False)

# ---------- Save ----------
os.makedirs("log", exist_ok=True)
output_path = f"log/metrics_eval_{args.tag}.json"
with open(output_path, "w") as f:
    json.dump({
        "model_dir": args.model_dir,
        "tag": args.tag,
        "model_file_size_MB": model_disk_size_mb,
        "model_type": "OpenVINO",
        "public": public_metrics,
        "private": private_metrics
    }, f, indent=2)

print(f"📊 Evaluation complete. Metrics saved to: {output_path}")


# import os 
# import json
# import time
# import argparse
# from tqdm import tqdm
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from evaluate import load

# # ---------- Parse Arguments ----------
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_dir", required=True, help="Path to model directory")
# parser.add_argument("--tag", default="latest", help="Tag to append to output file")
# parser.add_argument("--test_path", default="data/dataset/test.json", help="Public test set")
# parser.add_argument("--private_test_path", default="data/dataset/private_test.json", help="Private test set")
# args = parser.parse_args()

# # ---------- Load Model ----------
# print(f" Loading PyTorch model from {args.model_dir}")
# model = AutoModelForCausalLM.from_pretrained(args.model_dir)
# tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
# model_type = "PyTorch"

# # ---------- Compute Model Stats ----------
# def get_model_weight_file_size(model_dir):
#     total_bytes = 0
#     for fname in os.listdir(model_dir):
#         if fname.endswith((".bin", ".safetensors", ".pt")):
#             total_bytes += os.path.getsize(os.path.join(model_dir, fname))
#     return round(total_bytes / (1024 * 1024), 2)  # MB

# def get_model_dir_total_size(model_dir):
#     total_bytes = 0
#     for dirpath, _, filenames in os.walk(model_dir):
#         for f in filenames:
#             fp = os.path.join(dirpath, f)
#             total_bytes += os.path.getsize(fp)
#     return round(total_bytes / (1024 * 1024), 2)  # MB

# model_weight_file_mb = get_model_weight_file_size(args.model_dir)
# model_dir_total_size_mb = get_model_dir_total_size(args.model_dir)
# print(f"Model weight file size: {model_weight_file_mb} MB")
# print(f"Model directory total size: {model_dir_total_size_mb} MB")

# # ---------- Load Metrics ----------
# rouge = load("rouge")
# bleu = load("bleu")
# bertscore = load("bertscore")

# # ---------- Load Dataset ----------
# def load_data(path):
#     with open(path) as f:
#         return json.load(f)

# test_data = load_data(args.test_path)
# private_data = load_data(args.private_test_path)

# # ---------- Compute Perplexity ----------
# def compute_perplexity(texts):
#     ppl_list = []
#     for text in texts:
#         enc = tokenizer(text, return_tensors="pt")
#         with torch.no_grad():
#             output = model(**enc, labels=enc["input_ids"])
#             loss = output.loss
#         ppl = torch.exp(loss).item()
#         ppl_list.append(ppl)
#     return sum(ppl_list) / len(ppl_list) if ppl_list else float('nan')

# # ---------- Evaluation Function ----------
# def evaluate_dataset(dataset, has_answer=True):
#     predictions, references = [], []
#     em = 0
#     total_latency = 0
#     total_tokens = 0

#     for item in tqdm(dataset, desc="Evaluating"):
#         question = item["qa"]["question"]
#         expected = item["qa"]["answer"] if has_answer else None

#         inputs = tokenizer(question, return_tensors="pt")
#         start = time.time()
#         outputs = model.generate(**inputs, max_new_tokens=64)
#         latency = time.time() - start
#         total_latency += latency

#         decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         predictions.append(decoded)
#         if has_answer:
#             references.append(expected)
#             if decoded.strip().lower() == expected.strip().lower():
#                 em += 1

#         total_tokens += outputs[0].shape[0]

#     metrics = {}
#     if has_answer:
#         metrics["rougeL"] = rouge.compute(predictions=predictions, references=references)["rougeL"]
#         metrics["bleu"] = bleu.compute(predictions=predictions, references=references)["bleu"]
#         metrics["exact_match"] = round(em / len(dataset), 4)
#         metrics["bertscore_f1"] = sum(
#             bertscore.compute(predictions=predictions, references=references, lang="en")["f1"]
#         ) / len(predictions)
#         metrics["perplexity"] = compute_perplexity(predictions)

#     metrics["latency_sec"] = round(total_latency / len(dataset), 4)
#     metrics["throughput_tokens_per_sec"] = round(total_tokens / total_latency, 2)
#     return metrics

# # ---------- Run Evaluation ----------
# public_metrics = evaluate_dataset(test_data, has_answer=True)
# private_metrics = evaluate_dataset(private_data, has_answer=False)

# # ---------- Save ----------
# os.makedirs("log", exist_ok=True)
# output_path = f"log/metrics_eval_{args.tag}.json"
# with open(output_path, "w") as f:
#     json.dump({
#         "model_dir": args.model_dir,
#         "tag": args.tag,
#         "model_type": model_type,
#         "model_weight_file_MB": model_weight_file_mb,
#         "model_dir_total_size_MB": model_dir_total_size_mb,
#         "public": public_metrics,
#         "private": private_metrics
#     }, f, indent=2)

# print(f" Evaluation complete. Metrics saved to: {output_path}")