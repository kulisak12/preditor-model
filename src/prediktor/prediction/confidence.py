import re

import torch

from prediktor.config import Config
from prediktor.model import device, model, tokenizer

TERMINATORS = ".!?:;"


def predict(text: str) -> str:
    """Generate continuation for the given text."""
    # the model doesn't like whitespace at the after
    trimmed_text = text.rstrip()
    stripped_suffix = text[len(trimmed_text):]
    prediction = generate(trimmed_text, Config.confidence)
    if stripped_suffix:
        prediction = prediction.lstrip()

    # only use one sentence
    return first_sentence(prediction)


def first_sentence(text: str) -> str:
    """Return the first sentence of the given text."""
    first_terminator = find_first_occurence(text, TERMINATORS)
    if first_terminator != -1:
        return text[:first_terminator + 1]
    return text


def find_first_occurence(text: str, chars: str) -> int:
    """Return the index of the first occurence of any of the given chars."""
    pattern = f"[{re.escape(chars)}]"
    match = re.search(pattern, text)
    return match.start() if match else -1


def generate(input: str, confidence: float) -> str:
    """Use the model to generate a continuation of the input text.

    Uses top-k sampling.
    The higher the confidence, the longer the generated text.
    """
    # batch size is always 1
    input_ids = tokenizer.encode(input, return_tensors="pt")[0].to(device)
    original_length = input_ids.size(0)
    max_total_length = input_ids.size(0) + Config.max_length

    with torch.no_grad():
        while input_ids.size(0) < max_total_length:
            # take the last token
            logits = model(input_ids).logits[-1, :]
            # only sample from the top k tokens
            top_k = torch.topk(logits, k=Config.top_k)
            probabilities = torch.softmax(
                top_k.values / Config.temperature, dim=-1
            )
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
