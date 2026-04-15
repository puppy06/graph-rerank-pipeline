"""Persistent vector index for RAG (Chroma + HNSW cosine)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from providers.base import BaseModelProvider


@dataclass
class RetrievedChunk:
    """One passage returned by approximate nearest-neighbor search."""

    id: str
    text: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    distance: float


class ChromaChunkStore:
    """Store chunk embeddings in a local ChromaDB collection."""

    def __init__(self, persist_directory: str | Path, collection_name: str) -> None:
        import chromadb

        self._name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection (empty index)."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())

    def add(
        self,
        *,
        ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(texts) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, texts, metadatas, and embeddings length must match")
        embs = np.asarray(embeddings, dtype=np.float32)
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embs.tolist(),
            metadatas=metadatas,
        )

    def query(
        self, query_embedding: np.ndarray, n_results: int
    ) -> list[RetrievedChunk]:
        """Return up to n_results nearest chunks (by cosine distance in Chroma)."""
        total = self.count()
        if total == 0:
            return []
        n = min(max(1, int(n_results)), total)

        q = np.asarray(query_embedding, dtype=np.float32)
        raw = self._collection.query(
            query_embeddings=[q.tolist()],
            n_results=n,
            include=["documents", "embeddings", "distances", "metadatas"],
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        embs = np.asarray(raw["embeddings"][0], dtype=np.float32)
        dists = raw["distances"][0]
        metas = raw["metadatas"][0]

        out: list[RetrievedChunk] = []
        for i in range(len(ids)):
            meta = metas[i] or {}
            out.append(
                RetrievedChunk(
                    id=ids[i],
                    text=docs[i],
                    embedding=embs[i],
                    metadata=dict(meta),
                    distance=float(dists[i]),
                )
            )
        return out


def embed_batches(
    provider: BaseModelProvider,
    texts: list[str],
    *,
    batch_size: int,
    for_documents: bool,
) -> np.ndarray:
    """Run embedding in batches using document or query endpoints."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embed_fn = (
        provider.embed_documents if for_documents else provider.embed_query
    )
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        parts.append(embed_fn(batch))
    return np.vstack(parts)
