import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from preditor.model.model import Model


class HFModel(Model):
    def __init__(self, model_path: str, max_new_tokens: int):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._prefix_space_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            add_prefix_space=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self._config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id
        )

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def prefix_space_tokenizer(self) -> PreTrainedTokenizer:
        return self._prefix_space_tokenizer

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def config(self) -> GenerationConfig:
        return self._config
