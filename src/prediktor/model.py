import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from prediktor.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.model_path)
tokenizer_with_prefix = AutoTokenizer.from_pretrained(Config.model_path, use_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(Config.model_path)
generation_config = GenerationConfig(
    max_new_tokens=Config.max_length,
    pad_token_id=tokenizer.eos_token_id,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)
