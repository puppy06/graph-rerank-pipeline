"""Run the LangGraph-orchestrated RAG workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents import run_rag_graph
from config import RAG_CHROMA_PATH, RAG_COLLECTION_NAME
from data_pipeline import ChromaChunkStore
from providers import get_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph RAG: retrieve -> rerank -> synthesize/fallback."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--chroma-path", default=RAG_CHROMA_PATH)
    parser.add_argument("--collection", default=RAG_COLLECTION_NAME)
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--mode", choices=["cosine", "dot"], default="cosine")
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ChromaChunkStore(args.chroma_path, args.collection)
    if store.count() == 0:
        print(
            "Vector index is empty. Run ingest first, e.g.\n"
            "  python scripts/rag.py ingest --data-dir corpus --reset"
        )
        sys.exit(1)

    provider = get_provider()
    result = run_rag_graph(
        query=args.query,
        provider=provider,
        store=store,
        mode=args.mode,
        recall_k=args.recall_k,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        min_score=args.min_score,
        skip_generate=args.skip_generate,
    )

    print("Retrieved (post-rerank):")
    for idx, chunk in enumerate(result.reranked, start=1):
        source = chunk.metadata.get("source", "?")
        score = result.scores[idx - 1] if idx - 1 < len(result.scores) else float("nan")
        print(f"{idx}. score={score:.4f} [{chunk.id}] ({source})")
        #print(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
        print()

    if args.skip_generate:
        return

    if result.low_confidence:
        print("Low-confidence fallback:")
    else:
        print("Generated answer:")
    print(result.answer)


if __name__ == "__main__":
    main()
