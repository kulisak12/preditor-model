import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.path.join("models", "gpt2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, pad_token='<|endoftext|>')
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

MAX_WORDS = 10
TERMINATORS = ".!?:;"


def predict(text: str) -> str:
    # the model doesn't like whitespace at the end
    trimmed_text = text.rstrip()
    stripped_suffix = text[len(trimmed_text):]

    input_ids = tokenizer.encode(trimmed_text, return_tensors="pt")
    output_tokens = model.generate(
        input_ids,
        max_new_tokens=MAX_WORDS,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    output = tokenizer.decode(output_tokens[0])

    # remove input text
    prediction = output[len(trimmed_text):]
    if stripped_suffix:
        prediction = prediction.lstrip()
    # only use one sentence
    first_terminator = find_first_occurence(prediction, TERMINATORS)
    if first_terminator != -1:
        prediction = prediction[:first_terminator + 1]

    return prediction


def find_first_occurence(text: str, chars: str) -> int:
    pattern = f"[{re.escape(chars)}]"
    match = re.search(pattern, text)
    return match.start() if match else -1
