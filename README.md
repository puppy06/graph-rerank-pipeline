# Graph Rerank Pipeline

A multi-hop retrieval and reasoning demo built around **LangGraph** orchestration, **Cohere** embeddings and language models, and a **JAX** similarity layer for efficient re-ranking over retrieved passages.

The aim is to handle complex queries that need several retrieval steps (for example, comparing metrics across entities) while keeping the scoring path fast and explicit.

## Features (current and planned)

| Area | Status |
|------|--------|
| JAX dot and cosine similarity over a query vector and document matrix | Implemented in `math_ops/reranker.py` |
| LangGraph orchestration (retrieve, re-rank, synthesize, fallback) | Implemented in `agents/rag_graph.py` |
| Chroma vector store, chunking, and ingestion | Implemented under `data_pipeline/` |

## Repository layout

```
graph-rerank-pipeline/
├── agents/           # LangGraph StateGraph and node logic
├── corpus/           # Example documents for RAG ingest
├── data_pipeline/    # Ingestion, chunking, vector store
├── math_ops/         # JAX re-ranking and related numerics
│   └── reranker.py
├── providers/        # Cloud/local model provider implementations
├── scripts/          # CLI demos (hybrid rerank, RAG)
├── config.py         # Backend toggle and model settings
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

3. Create a `.env` file in the project root:

   ```bash
   USE_LOCAL_MODEL=false
   COHERE_API_KEY=your_key_here
   ```

   The project uses `python-dotenv` so applications can load these variables with `load_dotenv()`.

## Quick run demo

Run a complete flow (embed -> rerank -> generate) with whichever backend is selected:

```bash
python scripts/demo_hybrid_rerank.py
```

Rerank only (no generation):

```bash
python scripts/demo_hybrid_rerank.py --skip-generate
```

Custom query and docs:

```bash
python scripts/demo_hybrid_rerank.py \
  --query "Compare NVIDIA and AMD Q4 gross margins" \
  --doc "NVIDIA Q4 gross margin was 76.0%." \
  --doc "AMD Q4 gross margin was 51.0%." \
  --doc "Random unrelated paragraph." \
  --top-k 2
```

## RAG (ingest + retrieve + rerank + generate)

1. **Ingest** all `.txt` / `.md` files under a directory into a local **Chroma** index (embeddings use `embed_documents`). Use `--reset` to replace the collection.

   ```bash
   python scripts/rag.py ingest --data-dir corpus --reset
   ```

2. **Ask**: embed the question with `embed_query`, pull `recall-k` neighbors from Chroma, **rerank** every candidate with JAX (`math_ops/reranker.py`), then pass the top passages to the LLM.

   ```bash
   python scripts/rag.py ask --query "Compare NVIDIA and AMD Q4 gross margins."
   ```

   Optional env vars: `RAG_CHROMA_PATH` (default `.chroma`), `RAG_COLLECTION_NAME`, `RAG_EMBED_BATCH_SIZE`. The index directory is gitignored.

## LangGraph orchestration

Use a stateful LangGraph workflow for `retrieve -> rerank -> synthesize` with a
score-based fallback when retrieval confidence is low:

```bash
python scripts/langgraph_rag.py --query "Compare NVIDIA and AMD Q4 gross margins."
```

Useful options:
- `--skip-generate` prints reranked passages only.
- `--min-score` controls fallback threshold for top rerank score (default `0.2`).
- `--recall-k` controls ANN candidates from Chroma before reranking.

## Run in Google Colab (T4)

If you are using `colab_t4_gpu_setup.ipynb`, run the notebook directly in Colab, not from your local terminal.

1. Upload `colab_t4_gpu_setup.ipynb` to Google Drive.
2. In Drive, right-click the file and choose **Open with -> Google Colaboratory**.
3. In Colab, set **Runtime -> Change runtime type -> GPU**.
4. Run the notebook cells top to bottom.

Important:
- Run commands as Colab cells (for example, `!python scripts/rag.py ...`) after `%cd /content/graph-rerank-pipeline`.
- Do not run those same commands in your local PowerShell terminal if you expect Colab's T4 GPU to be used.

## Dual backend (one stack per run)

The codebase supports **two complete backends**; you pick one with **`USE_LOCAL_MODEL`**. Only that backend is constructed and used—Cohere and local Llama/BGE do not run together or share a mixed pipeline.

- `providers/base.py` defines `BaseModelProvider` with:
  - `embed(texts: list[str]) -> np.ndarray` (alias for document embeddings)
  - `embed_documents` / `embed_query` for RAG (Cohere uses `search_document` vs `search_query`)
  - `generate(prompt: str, *, max_new_tokens: int = 256) -> str`
- `providers/cohere_client.py` — full Cohere (`USE_LOCAL_MODEL=false`): embed + chat via the API.
- `providers/local_client.py` — full local (`USE_LOCAL_MODEL=true`): BGE-M3 + Llama 3 via `transformers` / `bitsandbytes`.
- `providers/factory.py` — `get_provider()` returns **either** a `CohereProvider` **or** a `LocalProvider`.

Set `USE_LOCAL_MODEL=true` for the open-source stack (e.g. Colab T4); keep it `false` and set `COHERE_API_KEY` for the cloud stack.

With Cohere, embeddings default to `embed-v4.0` and `COHERE_EMBED_API=auto` tries v2 then v1. Optional `COHERE_EMBED_FALLBACK_MODELS` lists extra model ids if the primary returns 404. Chat defaults to **`command-r-08-2024`** (`COHERE_GENERATE_MODEL`); older ids like `command-r-plus` were retired in Sept 2025—see [Cohere deprecations](https://docs.cohere.com/docs/deprecations). If your key has no Embed access, switch to **`USE_LOCAL_MODEL=true`** (full local) or use a production key with Embed enabled.

For GPU memory sharing with JAX, local mode sets:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Example usage:

```python
import jax.numpy as jnp
from providers import get_provider
from math_ops import rerank_indices

provider = get_provider()

query_vec = provider.embed_query(["What were NVIDIA and AMD Q4 gross margins?"])[0]
doc_matrix = provider.embed_documents(
    [
        "NVIDIA Q4 gross margin was 76.0%.",
        "AMD Q4 gross margin was 51.0%.",
        "A third unrelated passage.",
    ]
)

ranked_idx = rerank_indices(
    query=jnp.asarray(query_vec),
    doc_matrix=jnp.asarray(doc_matrix),
    mode="cosine",
    top_k=2,
)

answer = provider.generate("Compare NVIDIA and AMD Q4 gross margins.")
print(ranked_idx, answer)
```


## JAX re-ranking module

`math_ops/reranker.py` exposes JIT-compiled scoring and a helper to sort document indices:

- `dot_scores(query, doc_matrix)` unnormalized dot products, shape `(num_docs,)`.
- `cosine_scores(query, doc_matrix)` L2-normalized cosine similarity per row.
- `rerank_indices(query, doc_matrix, mode="cosine", top_k=None)` returns indices from best to worst match.

Embeddings from Cohere are typically `float32`; keep dtypes consistent with your JAX arrays for predictable behavior.

Run Python from the repository root so `import math_ops` resolves, or add the root to `PYTHONPATH`.

## License

Add a license file if you open-source this repository.
