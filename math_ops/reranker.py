"""
Batch similarity scoring between one query vector and many document vectors.

Core routines are JIT-compiled for repeated calls with the same array shapes.
Prefer float32 for broad hardware support; Cohere embeddings are typically float32.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

ScoreMode = Literal["dot", "cosine"]


@jax.jit
def dot_scores(query: Array, doc_matrix: Array) -> Array:
    """
    Unnormalized dot products: score[i] = dot(query, doc_matrix[i]).

    Args:
        query: Shape (dim,).
        doc_matrix: Shape (num_docs, dim), row-major document embeddings.

    Returns:
        Scores of shape (num_docs,).
    """
    q = jnp.asarray(query)
    d = jnp.asarray(doc_matrix)
    return d @ q


@jax.jit
def cosine_scores(query: Array, doc_matrix: Array) -> Array:
    """
    Cosine similarity after L2 normalization of the query and each document row.

    Equivalent to the dot product of unit vectors, in [-1, 1] for real inputs.

    Args:
        query: Shape (dim,).
        doc_matrix: Shape (num_docs, dim).

    Returns:
        Cosine similarities of shape (num_docs,).
    """
    q = jnp.asarray(query)
    d = jnp.asarray(doc_matrix)

    q_norm = jnp.linalg.norm(q)
    q_unit = jnp.where(q_norm > 0, q / q_norm, q)

    row_norms = jnp.linalg.norm(d, axis=1, keepdims=True)
    d_unit = jnp.where(row_norms > 0, d / row_norms, d)

    return (d_unit @ q_unit).ravel()


def rerank_indices(
    query: Array,
    doc_matrix: Array,
    mode: ScoreMode = "cosine",
    *,
    top_k: int | None = None,
) -> Array:
    """
    Return document indices sorted by descending similarity score.

    Underlying score functions are JIT-compiled; this wrapper handles mode and top_k.

    Args:
        query: Shape (dim,).
        doc_matrix: Shape (num_docs, dim).
        mode: "dot" or "cosine".
        top_k: If set, return only the top_k indices (best first).

    Returns:
        Integer indices of shape (num_docs,) or (top_k,).
    """
    if query.ndim != 1:
        raise ValueError("query must have shape (dim,)")
    if doc_matrix.ndim != 2:
        raise ValueError("doc_matrix must have shape (num_docs, dim)")
    if doc_matrix.shape[1] != query.shape[0]:
        raise ValueError("query dim must match doc_matrix second axis")

    if mode == "cosine":
        scores = cosine_scores(query, doc_matrix)
    else:
        scores = dot_scores(query, doc_matrix)

    order = jnp.argsort(scores)[::-1]
    if top_k is not None:
        return order[: int(top_k)]
    return order
