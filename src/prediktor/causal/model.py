import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prediktor.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.model_path)
tokenizer_with_prefix = AutoTokenizer.from_pretrained(Config.model_path, use_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(Config.model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)
