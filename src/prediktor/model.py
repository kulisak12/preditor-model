import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.path.join("models", "gpt2")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, pad_token='<|endoftext|>'
)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

MAX_NEW_LENGTH = 30
TOP_K = 10
TEMPERATURE = 0.7


def generate(input: str, confidence: float) -> str:
    """Use the model to generate a continuation of the input text.

    Uses top-k sampling.
    The higher the confidence, the longer the generated text.
    """
    # batch size is always 1
    input_ids = tokenizer.encode(input, return_tensors="pt")[0]
    original_length = input_ids.size(0)
    max_total_length = input_ids.size(0) + MAX_NEW_LENGTH

    with torch.no_grad():
        while input_ids.size(0) < max_total_length:
            # take the last token
            logits = model(input_ids).logits[-1, :]
            # only sample from the top k tokens
            top_k = torch.topk(logits, k=TOP_K)
            probabilities = torch.softmax(top_k.values / TEMPERATURE, dim=-1)
            # stop generation if probabilities are low
            confidence -= confidence_loss(probabilities)
            if confidence < 0:
                break

            next_token_index = torch.multinomial(probabilities, num_samples=1)
            next_token = top_k.indices[next_token_index]
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat((input_ids, next_token), dim=-1)

    generated_ids = input_ids[original_length:]
    generated_text = tokenizer.decode(
        generated_ids.squeeze().tolist(), skip_special_tokens=True
    )
    return generated_text


def confidence_loss(probabilities: torch.Tensor) -> float:
    """Estimate how much the confidence of the generated text decreases."""
    prob_sum = probabilities[:3].sum().item()
    return 1 - prob_sum**2
