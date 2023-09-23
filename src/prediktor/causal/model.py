from typing import Any, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prediktor.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.model_path)
tokenizer_with_prefix = AutoTokenizer.from_pretrained(Config.model_path, use_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(Config.model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)


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


def beam_search(
    input_text: str,
    bad_words: Iterable[str],
    end: str = "",
) -> List[str]:
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    num_end_tokens = len(tokenizer.encode(end, add_special_tokens=False))
    gen_ids = model.generate(
        input_ids,
        bad_words_ids=get_bad_tokens(bad_words),
        max_new_tokens=num_end_tokens + Config.max_length,
        num_return_sequences=Config.num_beams,
        num_beams=Config.num_beams,
        num_beam_groups=(Config.num_beams + 1) // 2,
        diversity_penalty=20.0,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return decoded_texts


def get_bad_tokens(bad_words: Iterable[str]) -> List[Any]:
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in bad_words:
        # some tokenizers accept the prefix space, some need the parameter
        tokenized_word = tokenizer_with_prefix([" " + word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list
