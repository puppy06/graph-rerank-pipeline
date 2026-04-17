"""RAG CLI: ingest a text corpus into Chroma, then retrieve + JAX rerank + generate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import numpy as np

from config import RAG_CHROMA_PATH, RAG_COLLECTION_NAME, RAG_EMBED_BATCH_SIZE
from data_pipeline import ChromaChunkStore, ingest_directory
from math_ops import rerank_indices
from providers import get_provider


def _cmd_ingest(args: argparse.Namespace) -> None:
    store = ChromaChunkStore(args.chroma_path, args.collection)
    provider = get_provider()
    batch = args.embed_batch_size or RAG_EMBED_BATCH_SIZE
    n = ingest_directory(
        data_dir=Path(args.data_dir),
        store=store,
        provider=provider,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_batch_size=batch,
        reset=args.reset,
    )
    print(f"Ingested {n} chunks into {args.chroma_path!s} (collection {args.collection!r}).")


def _cmd_ask(args: argparse.Namespace) -> None:
    store = ChromaChunkStore(args.chroma_path, args.collection)
    if store.count() == 0:
        print(
            "Vector index is empty. Run ingest first, e.g.\n"
            f"  python scripts/rag.py ingest --data-dir <path> --reset"
        )
        sys.exit(1)

    provider = get_provider()
    query_vec = provider.embed_query([args.query])[0]
    candidates = store.query(np.asarray(query_vec, dtype=np.float32), args.recall_k)
    if not candidates:
        print("No candidates returned from the index.")
        sys.exit(1)

    doc_matrix = np.stack([c.embedding for c in candidates], axis=0)
    ranked = rerank_indices(
        query=jnp.asarray(query_vec, dtype=jnp.float32),
        doc_matrix=jnp.asarray(doc_matrix, dtype=jnp.float32),
        mode=args.mode,
        top_k=None,
    )
    ranked_list = [int(i) for i in ranked.tolist()]
    ordered = [candidates[i] for i in ranked_list]

    print("Retrieved passages (after JAX rerank, best first):")
    for rank, ch in enumerate(ordered, start=1):
        src = ch.metadata.get("source", "?")
        print(f"{rank}. [{ch.id}] ({src})")
        # Uncomment to print the text of the chunk
        #print(ch.text[:500] + ("…" if len(ch.text) > 500 else ""))
        print()

    if args.skip_generate:
        return

    context_lines = []
    for ch in ordered[: args.top_k]:
        context_lines.append(f"- ({ch.metadata.get('source', '?')}) {ch.text}")
    context = "\n".join(context_lines)
    prompt = (
        "Answer the user query using only the provided context. "
        "If the context does not contain the answer, say you do not know.\n"
        f"User query: {args.query}\n"
        f"Context:\n{context}\n"
        "Answer:"
    )
    answer = provider.generate(prompt, max_new_tokens=args.max_new_tokens)
    print("Generated answer:")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG: ingest corpus and query with rerank.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_in = sub.add_parser("ingest", help="Chunk, embed, and index .txt / .md under a directory.")
    p_in.add_argument(
        "--data-dir",
        required=True,
        help="Root directory of text/markdown files (recursive).",
    )
    p_in.add_argument(
        "--chroma-path",
        default=RAG_CHROMA_PATH,
        help="Chroma persistence directory.",
    )
    p_in.add_argument(
        "--collection",
        default=RAG_COLLECTION_NAME,
        help="Chroma collection name.",
    )
    p_in.add_argument("--chunk-size", type=int, default=1200)
    p_in.add_argument("--chunk-overlap", type=int, default=200)
    p_in.add_argument(
        "--embed-batch-size",
        type=int,
        default=0,
        help=f"0 = use config default ({RAG_EMBED_BATCH_SIZE}).",
    )
    p_in.add_argument(
        "--reset",
        action="store_true",
        help="Drop the collection before ingesting.",
    )
    p_in.set_defaults(func=_cmd_ingest)

    p_q = sub.add_parser("ask", help="Query the index, rerank with JAX, optionally generate.")
    p_q.add_argument("--query", required=True)
    p_q.add_argument("--chroma-path", default=RAG_CHROMA_PATH)
    p_q.add_argument("--collection", default=RAG_COLLECTION_NAME)
    p_q.add_argument(
        "--recall-k",
        type=int,
        default=20,
        help="ANN candidates from Chroma before JAX rerank.",
    )
    p_q.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Passages to include in the generation prompt (after rerank).",
    )
    p_q.add_argument("--mode", choices=["cosine", "dot"], default="cosine")
    p_q.add_argument("--max-new-tokens", type=int, default=256)
    p_q.add_argument("--skip-generate", action="store_true")
    p_q.set_defaults(func=_cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
