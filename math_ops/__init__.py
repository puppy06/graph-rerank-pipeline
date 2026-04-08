"""JAX-based similarity and re-ranking utilities."""

from .reranker import cosine_scores, dot_scores, rerank_indices

__all__ = ["cosine_scores", "dot_scores", "rerank_indices"]
