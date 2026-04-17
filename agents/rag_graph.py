"""LangGraph orchestration for retrieval -> rerank -> synthesize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import jax.numpy as jnp
import numpy as np
from langgraph.graph import END, START, StateGraph

from data_pipeline.vector_store import ChromaChunkStore, RetrievedChunk
from math_ops import cosine_scores, dot_scores, rerank_indices
from providers.base import BaseModelProvider


@dataclass
class GraphRunResult:
    """Structured output from a LangGraph RAG run."""

    answer: str
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    scores: list[float]
    low_confidence: bool


class RagState(TypedDict):
    """Execution state passed between graph nodes."""

    query: str
    mode: str
    recall_k: int
    top_k: int
    max_new_tokens: int
    min_score: float
    skip_generate: bool
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    scores: list[float]
    low_confidence: bool
    answer: str


def _retrieve_node(
    state: RagState, *, provider: BaseModelProvider, store: ChromaChunkStore
) -> RagState:
    query_vec = provider.embed_query([state["query"]])[0]
    candidates = store.query(np.asarray(query_vec, dtype=np.float32), state["recall_k"])
    return {**state, "candidates": candidates}


def _rerank_node(state: RagState, *, provider: BaseModelProvider) -> RagState:
    if not state["candidates"]:
        return {**state, "reranked": [], "scores": [], "low_confidence": True}

    query_vec = provider.embed_query([state["query"]])[0]
    doc_matrix = np.stack([c.embedding for c in state["candidates"]], axis=0)

    ranked = rerank_indices(
        query=jnp.asarray(query_vec, dtype=jnp.float32),
        doc_matrix=jnp.asarray(doc_matrix, dtype=jnp.float32),
        mode=state["mode"],
        top_k=None,
    )
    ranked_idx = [int(i) for i in ranked.tolist()]
    ordered = [state["candidates"][i] for i in ranked_idx]

    if state["mode"] == "dot":
        score_arr = dot_scores(
            jnp.asarray(query_vec, dtype=jnp.float32),
            jnp.asarray(doc_matrix, dtype=jnp.float32),
        )
    else:
        score_arr = cosine_scores(
            jnp.asarray(query_vec, dtype=jnp.float32),
            jnp.asarray(doc_matrix, dtype=jnp.float32),
        )
    raw_scores = [float(x) for x in np.asarray(score_arr)[ranked_idx].tolist()]

    top_k = min(state["top_k"], len(ordered))
    trimmed = ordered[:top_k]
    trimmed_scores = raw_scores[:top_k]
    top_score = trimmed_scores[0] if trimmed_scores else -1.0
    low_conf = top_score < state["min_score"]
    return {
        **state,
        "reranked": trimmed,
        "scores": trimmed_scores,
        "low_confidence": low_conf,
    }


def _route_after_rerank(state: RagState) -> str:
    if state["skip_generate"]:
        return "done"
    if state["low_confidence"]:
        return "fallback"
    return "synthesize"


def _synthesize_node(state: RagState, *, provider: BaseModelProvider) -> RagState:
    context_lines = []
    for chunk in state["reranked"]:
        source = chunk.metadata.get("source", "?")
        context_lines.append(f"- ({source}) {chunk.text}")
    context = "\n".join(context_lines)
    prompt = (
        "Answer the user query using only the provided context. "
        "If the context does not contain the answer, say you do not know.\n"
        f"User query: {state['query']}\n"
        f"Context:\n{context}\n"
        "Answer:"
    )
    answer = provider.generate(prompt, max_new_tokens=state["max_new_tokens"])
    return {**state, "answer": answer}


def _fallback_node(state: RagState) -> RagState:
    msg = (
        "Insufficient retrieval confidence to generate a grounded answer. "
        "Try rephrasing the query, increasing recall_k, or ingesting more documents."
    )
    return {**state, "answer": msg}


def build_rag_graph(*, provider: BaseModelProvider, store: ChromaChunkStore) -> Any:
    """Build and compile the LangGraph state machine for RAG orchestration."""
    graph = StateGraph(RagState)
    graph.add_node("retrieve", lambda state: _retrieve_node(state, provider=provider, store=store))
    graph.add_node("rerank", lambda state: _rerank_node(state, provider=provider))
    graph.add_node("synthesize", lambda state: _synthesize_node(state, provider=provider))
    graph.add_node("fallback", _fallback_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        _route_after_rerank,
        {"synthesize": "synthesize", "fallback": "fallback", "done": END},
    )
    graph.add_edge("synthesize", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def run_rag_graph(
    *,
    query: str,
    provider: BaseModelProvider,
    store: ChromaChunkStore,
    mode: str = "cosine",
    recall_k: int = 20,
    top_k: int = 4,
    max_new_tokens: int = 256,
    min_score: float = 0.2,
    skip_generate: bool = False,
) -> GraphRunResult:
    """Execute the LangGraph workflow and return structured results."""
    app = build_rag_graph(provider=provider, store=store)
    initial: RagState = {
        "query": query,
        "mode": mode,
        "recall_k": int(recall_k),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "min_score": float(min_score),
        "skip_generate": bool(skip_generate),
        "candidates": [],
        "reranked": [],
        "scores": [],
        "low_confidence": False,
        "answer": "",
    }
    out = app.invoke(initial)
    return GraphRunResult(
        answer=out["answer"],
        candidates=out["candidates"],
        reranked=out["reranked"],
        scores=out["scores"],
        low_confidence=out["low_confidence"],
    )
