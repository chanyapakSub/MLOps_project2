import json
from torch.utils.data import Dataset


def load_finqa(path):
    """
    Load raw FinQA data from GitHub JSON files.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



class FinQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["qa"]["question"]
        # รวม pre_text + table + post_text เป็น context
        context = " ".join(example.get("pre_text", [])) + " " + \
                  " ".join(example.get("table", [])) + " " + \
                  " ".join(example.get("post_text", []))
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Note: ยังไม่มีตำแหน่ง start/end เพราะเราใช้ exe_ans (generative QA)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.tokenizer(example["qa"]["exe_ans"], 
                                     truncation=True,
                                     padding="max_length",
                                     max_length=self.max_length,
                                     return_tensors="pt")["input_ids"].squeeze(0)
        }
