import torch

from preditor.config import Config
from preditor.model.model import Model


def generate(model: Model, input_text: str) -> str:
    """Use the model.model to generate a continuation of the input text.

    Uses top-k sampling.
    The higher the confidence, the longer the generated text.
    """
    # batch size is always 1
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt")[0].to(model.device)
    original_length = input_ids.size(0)
    max_total_length = input_ids.size(0) + Config.max_length

    with torch.no_grad():
        confidence = Config.confidence
        while input_ids.size(0) < max_total_length:
            # take the last token
            logits = model.model(input_ids).logits[-1, :]
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
            if next_token.item() == model.tokenizer.eos_token_id:
                break
            input_ids = torch.cat((input_ids, next_token), dim=-1)

    output_ids = input_ids[original_length:]
    decoded_text = model.tokenizer.decode(
        output_ids.squeeze().tolist(), skip_special_tokens=True
    )
    return decoded_text


def confidence_loss(probabilities: torch.Tensor) -> float:
    """Estimate how much the confidence of the generated text decreases."""
    prob_sum = probabilities[:3].sum().item()
    return 1 - prob_sum**2
