"""This module provides the interface for a model."""

import abc

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer


class Model(abc.ABC):
    """The interface for a model."""

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
