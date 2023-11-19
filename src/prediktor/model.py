import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from prediktor.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.model_path)
tokenizer_with_prefix = AutoTokenizer.from_pretrained(Config.model_path, use_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(
    Config.model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
generation_config = GenerationConfig(
    max_new_tokens=Config.max_length,
    pad_token_id=tokenizer.eos_token_id,
)
device = model.device


def encode_with_eos(text: str) -> torch.Tensor:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    eos_tensor = torch.tensor(tokenizer.eos_token_id).reshape(1, 1)
    return torch.cat([eos_tensor, input_ids], dim=-1)
