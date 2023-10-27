from prediktor import model
from prediktor.config import Config


def generate(input_text: str) -> str:
    """Use the model.model to generate a continuation of the input text."""
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output_ids = model.model.generate(
        input_ids,
        generation_config=model.generation_config
    )
    decoded_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_text
