from typing import List

from prediktor import model
from prediktor.config import Config

PROMPT = "Write a sentence such that it ends with:"


def infill_between(before_cursor: str, after_cursor: str) -> str:
    """Generate an infill between the given strings."""
    prompt = format_infill_prompt(before_cursor, after_cursor)
    decoded = beam_search(prompt, after_cursor)
    outputs = [extract_output(text, before_cursor) for text in decoded]
    best_output = get_best_output(outputs, after_cursor).rstrip()
    return best_output


def format_infill_prompt(before: str, after: str) -> str:
    """Create the prompt for the infill generation."""
    return f"{PROMPT} {after}\n{before}"


def extract_output(text: str, before_cursor: str) -> str:
    filled = text[text.find(before_cursor) + len(before_cursor):]
    return filled[:filled.find("\n")]


def get_best_output(outputs: List[str], after: str) -> str:
    """Return the output that best matches the text after cursor.

    Favor outputs in the beginning of the list.
    """
    # exact match
    for output in outputs:
        if output.endswith(after):
            return output[:len(output) - len(after)]
    # partial match
    end_first_word = after.split()[0]
    for output in outputs:
        pos = output.find(end_first_word)
        if pos != -1:
            return output[:pos]
    # no match
    for output in outputs:
        splits = output.split()
        if len(splits) > 1:
            return splits[0]
    # no output
    return ""


def beam_search(
    input_text: str,
    end: str = "",
) -> List[str]:
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    num_end_tokens = len(model.tokenizer.encode(end, add_special_tokens=False))
    gen_ids = model.model.generate(
        input_ids,
        max_new_tokens=num_end_tokens + Config.max_length,
        num_return_sequences=Config.num_beams,
        num_beams=Config.num_beams,
        num_beam_groups=(Config.num_beams + 1) // 2,
        diversity_penalty=20.0,
        pad_token_id=model.tokenizer.eos_token_id
    )
    decoded_texts = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return decoded_texts
