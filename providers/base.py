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
        Defaults to document embeddings; prefer embed_documents / embed_query for RAG.
        """

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embeddings for passages stored in the vector index."""
        return self.embed(texts)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        """Embeddings for user queries (retrieval). Defaults to same as embed_documents."""
        return self.embed(texts)

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        """Generate a text response for the provided prompt."""
