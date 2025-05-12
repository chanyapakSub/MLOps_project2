from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"
SAVE_DIR = "models/deepseek-7b-chat"

print("Loading DeepSeek LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # ใช้ float16 เพื่อความเบาแต่ไม่ลดคุณภาพมาก
    device_map="auto"
)
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)




