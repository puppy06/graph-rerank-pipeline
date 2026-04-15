"""Load files from disk, chunk, embed, and write to the vector store."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from data_pipeline.chunking import chunk_text
from data_pipeline.vector_store import ChromaChunkStore, embed_batches
from providers.base import BaseModelProvider

_TEXT_SUFFIXES = {".txt", ".md"}


def _chunk_id(relative_path: str, chunk_index: int) -> str:
    digest = hashlib.sha256(
        f"{relative_path}\n{chunk_index}".encode("utf-8")
    ).hexdigest()[:32]
    return f"c_{digest}"


def iter_corpus_files(root: Path) -> list[Path]:
    """Sorted list of .txt / .md files under root (recursive)."""
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
            files.append(path)
    return files


def ingest_directory(
    *,
    data_dir: Path,
    store: ChromaChunkStore,
    provider: BaseModelProvider,
    chunk_size: int,
    chunk_overlap: int,
    embed_batch_size: int,
    reset: bool = False,
) -> int:
    """
    Chunk all text/markdown files under data_dir, embed, and add to store.

    Returns:
        Number of chunks indexed.
    """
    if reset:
        store.reset()

    files = iter_corpus_files(data_dir)
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for file_path in files:
        rel = str(file_path.relative_to(data_dir)).replace("\\", "/")
        body = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(
            body, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for idx, chunk in enumerate(chunks):
            ids.append(_chunk_id(rel, idx))
            texts.append(chunk)
            metadatas.append(
                {
                    "source": rel,
                    "chunk_index": idx,
                }
            )

    if not texts:
        return 0

    embeddings = embed_batches(
        provider,
        texts,
        batch_size=embed_batch_size,
        for_documents=True,
    )
    store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
    return len(texts)
