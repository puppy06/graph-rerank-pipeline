"""Shared interface for model providers used by the retrieval pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModelProvider(ABC):
    """Abstract provider with embedding and generation capabilities."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Return dense embeddings for input texts.

        Implementations should return shape (num_texts, dim) float32 arrays.
        """

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        """Generate a text response for the provided prompt."""
