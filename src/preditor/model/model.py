import abc

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer


class Model(abc.ABC):
    @property
    @abc.abstractmethod
    def model(self) -> PreTrainedModel:
        pass

    @property
    @abc.abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer:
        pass

    @property
    @abc.abstractmethod
    def prefix_space_tokenizer(self) -> PreTrainedTokenizer:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abc.abstractmethod
    def config(self) -> GenerationConfig:
        pass
