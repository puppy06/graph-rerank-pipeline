# Graph Rerank Pipeline

A multi-hop retrieval and reasoning demo built around **LangGraph** orchestration, **Cohere** embeddings and language models, and a **JAX** similarity layer for efficient re-ranking over retrieved passages.

The aim is to handle complex queries that need several retrieval steps (for example, comparing metrics across entities) while keeping the scoring path fast and explicit.

## Features (current and planned)

| Area | Status |
|------|--------|
| JAX dot and cosine similarity over a query vector and document matrix | Implemented in `math_ops/reranker.py` |
| LangGraph agent (search, re-rank, verify, synthesize) | Planned under `agents/` |
| Local vector store (ChromaDB or Qdrant) and ingestion | Planned under `data_pipeline/` |

## Repository layout

```
graph-rerank-pipeline/
├── agents/           # LangGraph StateGraph and node logic
├── data_pipeline/    # Ingestion, chunking, vector store
├── math_ops/         # JAX re-ranking and related numerics
│   └── reranker.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer recommended
- A [Cohere](https://docs.cohere.com/) API key for embeddings, rerank, and Command when you wire the full agent

## Setup

1. Create and activate a virtual environment (recommended).

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root (do not commit secrets):

   ```bash
   COHERE_API_KEY=your_key_here
   ```

   The project uses `python-dotenv` so applications can load these variables with `load_dotenv()`.

## JAX re-ranking module

`math_ops/reranker.py` exposes JIT-compiled scoring and a helper to sort document indices:

- `dot_scores(query, doc_matrix)` unnormalized dot products, shape `(num_docs,)`.
- `cosine_scores(query, doc_matrix)` L2-normalized cosine similarity per row.
- `rerank_indices(query, doc_matrix, mode="cosine", top_k=None)` returns indices from best to worst match.

Embeddings from Cohere Embed v3 are typically `float32`; keep dtypes consistent with your JAX arrays for predictable behavior.

Run Python from the repository root so `import math_ops` resolves, or add the root to `PYTHONPATH`.

## License

Add a license file if you open-source this repository.
