from preditor.model.model import Model
from preditor.prediction.config import PredictionConfig


def generate(model: Model, input_text: str, config: PredictionConfig) -> str:
    """Generate a continuation of the input text."""
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output_ids = model.model.generate(
        input_ids,
        generation_config=model.config,
        max_new_tokens=config.max_length,
    )
    gen_ids = output_ids[0][len(input_ids[0]):]
    decoded_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return decoded_text
