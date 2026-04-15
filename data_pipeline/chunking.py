"""Split long texts into overlapping chunks for embedding and retrieval."""

from __future__ import annotations


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into fixed-size character windows with overlap.

    Args:
        text: Full document body.
        chunk_size: Maximum characters per chunk (before overlap trim).
        chunk_overlap: Characters reused from the previous chunk start.

    Returns:
        Non-empty chunk strings in order.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[str] = []
    start = 0
    n = len(stripped)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(stripped[start:end])
        if end >= n:
            break
        start = end - chunk_overlap
    return chunks
