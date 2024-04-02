import torch

from preditor.config import Config
from preditor.server import model


def choose_continuation_manually(input_text: str) -> None:
    """Generate continuation, prompting the user to choose the next token."""
    # based on preditor.model.model.generate
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt")[0]

    with torch.no_grad():
        while True:
            print(model.tokenizer.decode(input_ids))
            # take the last token
            logits = model.model(input_ids).logits[-1, :]
            # only sample from the top k tokens
            top_k = torch.topk(logits, k=8)
            probabilities = torch.softmax(
                top_k.values / Config.temperature, dim=-1
            )

            for i, (token_id, prob) in enumerate(
                zip(top_k.indices, probabilities)
            ):
                print_option(i, token_id, prob)
            print()
            choice = int(input())
            next_token_index = torch.tensor([choice])
            next_token = top_k.indices[next_token_index]
            input_ids = torch.cat((input_ids, next_token), dim=-1)



def print_option(i, token_id, prob):
    token = model.tokenizer.decode(token_id.item())
    print(f"{[i]}: {token} ({prob.item():.3f})  ", end="")
