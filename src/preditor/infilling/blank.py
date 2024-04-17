from typing import Any, Iterable, List

from preditor.model.model import Model

PROMPT = """\
### Instruction:
{}

### Input:
{}

### Output:
{}"""
INFILL_INSTRUCTION = "Fill in the blank marked by [...]"

bad_words = ["[...]", "[", "...", "[…]", "…"]
bad_words.extend("#" * i for i in range(1, 5))
bad_words.extend(" " * i for i in range(2, 10))
bad_words.extend("_" * i for i in range(2, 10))


def generate_infills(
    model: Model, before_cursor: str, after_cursor: str,
    max_length: int, num_variants: int
) -> List[str]:
    """Generate possible infills between the given strings."""
    had_trailing_space = bool(before_cursor) and before_cursor[-1].isspace()
    before_cursor = before_cursor.rstrip()
    after_cursor = after_cursor.lstrip()
    prompt = format_infill_prompt(before_cursor, after_cursor)
    decoded = beam_search(
        model, prompt, bad_words, after_cursor,
        max_length, num_variants
    )
    outputs = [extract_output(text, before_cursor) for text in decoded]
    outputs = [output.rstrip() for output in outputs]
    if had_trailing_space:
        outputs = [output.lstrip() for output in outputs]
    return outputs


def format_infill_prompt(before: str, after: str) -> str:
    """Create the prompt for the infill generation."""
    return PROMPT.format(
        INFILL_INSTRUCTION,
        before + " [...] " + after,
        before
    )


def extract_output(text: str, before_cursor: str) -> str:
    start = "### Output:\n" + before_cursor
    filled = text[text.find(start) + len(start):]
    return filled[:filled.find("\n")]


def beam_search(
    model: Model,
    input_text: str,
    bad_words: Iterable[str],
    end: str,
    max_length: int,
    num_variants: int
) -> List[str]:
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    num_end_tokens = len(model.tokenizer.encode(end, add_special_tokens=False))
    gen_ids = model.model.generate(
        input_ids,
        bad_words_ids=get_bad_tokens(model, bad_words),
        max_new_tokens=num_end_tokens + max_length,
        num_return_sequences=num_variants,
        num_beams=num_variants,
        num_beam_groups=(num_variants + 1) // 2,
        diversity_penalty=20.0,
        pad_token_id=model.tokenizer.eos_token_id
    )
    decoded_texts = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return decoded_texts


def get_bad_tokens(model: Model, bad_words: Iterable[str]) -> List[Any]:
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in bad_words:
        # some model.tokenizers accept the prefix space,
        # some need the special tokenizer
        tokenized_word = model.prefix_space_tokenizer(
            [" " + word], add_special_tokens=False
        ).input_ids[0]
        tokens_list.append(tokenized_word)
        tokenized_word = model.tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list
