# import os
# import json
# import torch
# from torch.utils.data import Dataset
# from dataclasses import dataclass
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling,
# )

# # ===== LOAD CONFIG =====
# with open("configs/train_config.json") as f:
#     train_cfg = json.load(f)

# with open("configs/model_config.json") as f:
#     model_cfg = json.load(f)

# # ===== CUSTOM DATASET =====
# class PromptDataset(Dataset):
#     def __init__(self, path, tokenizer, max_length=512):
#         self.data = []
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 item = json.loads(line)
#                 prompt = item["prompt"]
#                 answer = item["response"]
#                 full = prompt + " " + answer
#                 tokenized = tokenizer(
#                     full,
#                     truncation=True,
#                     padding="max_length",
#                     max_length=max_length
#                 )
#                 tokenized["labels"] = tokenized["input_ids"].copy()
#                 self.data.append(tokenized)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return {k: torch.tensor(v) for k, v in self.data[idx].items()}

# # ===== LOAD TOKENIZER & MODEL =====
# tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"])
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_cfg["model_path"],
#     torch_dtype=torch.float16 if train_cfg["fp16"] else torch.float32,
#     device_map="auto"
# )

# # ===== LOAD DATA =====
# train_dataset = PromptDataset(model_cfg["train_file"], tokenizer, max_length=train_cfg["max_length"])
# eval_dataset = PromptDataset(model_cfg["eval_file"], tokenizer, max_length=train_cfg["max_length"])

# # ===== COLLATOR =====
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False
# )

# # ===== TRAINING ARGUMENTS =====
# training_args = TrainingArguments(
#     output_dir=model_cfg["output_dir"],
#     per_device_train_batch_size=train_cfg["train_batch_size"],
#     per_device_eval_batch_size=train_cfg["eval_batch_size"],
#     num_train_epochs=train_cfg["epochs"],
#     eval_strategy="epoch",  
#     save_strategy="epoch",
#     logging_steps=train_cfg["logging_steps"],
#     save_total_limit=train_cfg["save_total_limit"],
#     gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
#     learning_rate=train_cfg["learning_rate"],
#     fp16=False, 
#     overwrite_output_dir=True,
#     report_to="none"
# )


# # ===== TRAIN =====
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# trainer.train()
# trainer.save_model(model_cfg["output_dir"])
# tokenizer.save_pretrained(model_cfg["output_dir"])

# print(f"Training complete. Model saved to {model_cfg['output_dir']}")



import os
import json
import tarfile
import argparse
import boto3
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ===== PARSE ARGS or fallback to config =====
parser = argparse.ArgumentParser()
parser.add_argument("--model_s3_path", type=str)
parser.add_argument("--train_file", type=str)
parser.add_argument("--eval_file", type=str)
parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
parser.add_argument("--fp16", type=bool, default=False)
parser.add_argument("--upload_s3_path", type=str)
args, unknown = parser.parse_known_args()

if not any(vars(args).values()):
    with open("configs/model_config.json") as f:
        cfg = json.load(f)
    args.model_s3_path = cfg["model_s3_path"]
    args.train_file = cfg["train_file"]
    args.eval_file = cfg["eval_file"]
    args.output_dir = cfg["output_dir"]
    args.upload_s3_path = cfg["model_s3_path_finetuned"] + "model.tar.gz"
    args.fp16 = False

REGION = "ap-southeast-2"

# ===== DOWNLOAD FUNCTIONS =====
def download_model_from_s3(s3_uri, local_dir):
    s3 = boto3.client("s3", region_name=REGION)
    bucket, key_prefix = s3_uri.replace("s3://", "").split("/", 1)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            local_path = os.path.join(local_dir, os.path.relpath(s3_key, key_prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, s3_key, local_path)

def download_file_from_s3(s3_uri, local_path):
    s3 = boto3.client("s3", region_name=REGION)
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

# ===== UPLOAD TO S3 AFTER TRAINING =====
def upload_model_to_s3(local_dir, s3_uri):
    tar_path = "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_dir, arcname=".")
    s3 = boto3.client("s3", region_name=REGION)
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3.upload_file(tar_path, bucket, key)
    print(f"✅ Uploaded to s3://{bucket}/{key}")

# ===== CUSTOM DATASET =====
class PromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                full = item["prompt"] + " " + item["response"]
                tokenized = tokenizer(
                    full,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length
                )
                tokenized["labels"] = tokenized["input_ids"].copy()
                self.data.append(tokenized)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}

# ===== DOWNLOAD BASE MODEL TO LOCAL =====
print(f"📦 Downloading base model from: {args.model_s3_path}")
local_model_path = "./tmp_model"
download_model_from_s3(args.model_s3_path, local_model_path)

# ===== DOWNLOAD DATASET FILES TO LOCAL =====
local_train_path = "./tmp_data/train.jsonl"
local_eval_path = "./tmp_data/dev.jsonl"
print(f"📥 Downloading dataset files...")
download_file_from_s3(args.train_file, local_train_path)
download_file_from_s3(args.eval_file, local_eval_path)

# ===== LOAD TOKENIZER & MODEL =====
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16 if args.fp16 else torch.float32
)

# ===== LOAD DATA =====
train_dataset = PromptDataset(local_train_path, tokenizer)
eval_dataset = PromptDataset(local_eval_path, tokenizer)

# ===== COLLATOR =====
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== TRAINING ARGS =====
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    fp16=args.fp16,
    overwrite_output_dir=True,
    report_to="none"
)

# ===== TRAIN =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# ===== REMOVE BASE WEIGHT BEFORE UPLOAD =====
bin_path = os.path.join(args.output_dir, "pytorch_model.bin")
if os.path.exists(bin_path):
    print(f"🧹 Removing base model weight: {bin_path}")
    os.remove(bin_path)

# ===== UPLOAD TO S3 =====
upload_model_to_s3(args.output_dir, args.upload_s3_path)

print(f"✅ Training complete. Fine-tuned model uploaded to {args.upload_s3_path}")
