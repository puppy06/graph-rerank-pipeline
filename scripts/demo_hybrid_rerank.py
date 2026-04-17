"""Run embed -> JAX rerank -> generate using the backend selected in config (Cohere or local)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path so `math_ops` / `providers` resolve when run as
# `python scripts/demo_hybrid_rerank.py` or `cd scripts && python demo_hybrid_rerank.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp

from math_ops import rerank_indices
from providers import get_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed, rerank, and generate using USE_LOCAL_MODEL from config."
    )
    parser.add_argument(
        "--query",
        default="Compare NVIDIA and AMD Q4 gross margins.",
        help="Query to embed and rerank against documents.",
    )
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        default=[],
        help="Document candidate passage. Can be repeated.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top documents to keep.",
    )
    parser.add_argument(
        "--mode",
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity mode for JAX reranker.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max new tokens for provider.generate.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only run embedding + reranking without final generation.",
    )
    return parser.parse_args()


def default_docs() -> list[str]:
    #default_path = _ROOT / "docs" / "The Shifting Landscape of Semiconductor Margins.txt"
    #if default_path.exists():
    #    return [default_path.read_text(encoding="utf-8-sig", errors="replace").strip()]
    return [
        "NVIDIA reported a Q4 gross margin of 76.0%.",
        "AMD reported a Q4 gross margin of 51.0%.",
        "This paragraph discusses supply chain logistics and is unrelated.",
    ]


def main() -> None:
    args = parse_args()
    docs = args.docs if args.docs else default_docs()

    provider = get_provider()

    query_embedding = provider.embed_query([args.query])[0]
    doc_embeddings = provider.embed_documents(docs)

    ranked = rerank_indices(
        query=jnp.asarray(query_embedding),
        doc_matrix=jnp.asarray(doc_embeddings),
        mode=args.mode,
        top_k=args.top_k,
    )
    ranked_list = [int(i) for i in ranked.tolist()]

    print("Top documents (best first):")
    for rank, idx in enumerate(ranked_list, start=1):
        print(f"{rank}. [doc {idx}] {docs[idx]}")

    if args.skip_generate:
        return

    context = "\n".join(f"- {docs[idx]}" for idx in ranked_list)
    prompt = (
        "Answer the user query using the provided context only.\n"
        f"User query: {args.query}\n"
        f"Context:\n{context}\n"
        "Answer:"
    )
    answer = provider.generate(prompt, max_new_tokens=args.max_new_tokens)
    print("\nGenerated answer:")
    print(answer)


if __name__ == "__main__":
    main()
