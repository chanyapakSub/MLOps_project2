import os
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ===== LOAD CONFIG =====
with open("configs/train_config.json") as f:
    train_cfg = json.load(f)

with open("configs/model_config.json") as f:
    model_cfg = json.load(f)

# ===== CUSTOM DATASET =====
class PromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = item["prompt"]
                answer = item["response"]
                full = prompt + " " + answer
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

# ===== LOAD TOKENIZER & MODEL =====
tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_cfg["model_path"],
    torch_dtype=torch.float16 if train_cfg["fp16"] else torch.float32,
    device_map="auto"
)

# ===== LOAD DATA =====
train_dataset = PromptDataset(model_cfg["train_file"], tokenizer, max_length=train_cfg["max_length"])
eval_dataset = PromptDataset(model_cfg["eval_file"], tokenizer, max_length=train_cfg["max_length"])

# ===== COLLATOR =====
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ===== TRAINING ARGUMENTS =====
training_args = TrainingArguments(
    output_dir=model_cfg["output_dir"],
    per_device_train_batch_size=train_cfg["train_batch_size"],
    per_device_eval_batch_size=train_cfg["eval_batch_size"],
    num_train_epochs=train_cfg["epochs"],
    eval_strategy="epoch",  
    save_strategy="epoch",
    logging_steps=train_cfg["logging_steps"],
    save_total_limit=train_cfg["save_total_limit"],
    gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    learning_rate=train_cfg["learning_rate"],
    fp16=False, 
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
trainer.save_model(model_cfg["output_dir"])
tokenizer.save_pretrained(model_cfg["output_dir"])

print(f"Training complete. Model saved to {model_cfg['output_dir']}")
