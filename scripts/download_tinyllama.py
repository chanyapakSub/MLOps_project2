from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_DIR = "data/model/tinyllama-1.1b"

# ตรวจสอบว่ามี CUDA หรือไม่
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model to: {device}")

# โหลด tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ตั้งค่า pad_token ให้เท่ากับ eos_token (LLaMA ไม่มี pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# โหลด model เป็น float16 เพื่อประหยัด VRAM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",               # ให้ HuggingFace จัดการให้
    torch_dtype=torch.float16        # ประหยัด VRAM (8GB ก็รันได้)
)

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(SAVE_DIR, exist_ok=True)

# บันทึก model และ tokenizer
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model + tokenizer saved to: {SAVE_DIR}")
