"""Central runtime toggles for model backend selection."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Toggle backend:
# - False => Cohere API
# - True  => Local HF models (Llama 3 + BGE-M3)
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

# Cohere options
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
# Trial / v2 API: embed-v4.0 is usually available; embed-english-v3.0 often404s without full access.
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
# Only used for embed-v4.x (Matryoshka); ignored for older embed models.
COHERE_EMBED_OUTPUT_DIMENSION = int(os.getenv("COHERE_EMBED_OUTPUT_DIMENSION", "1024"))
# Trial keys often 404 on /v2/embed but succeed on /v1/embed — use auto to try both.
COHERE_EMBED_API = os.getenv("COHERE_EMBED_API", "auto").strip().lower()
# Comma-separated; tried in order if primary model returns 404 on both APIs (optional).
_fallback_raw = os.getenv("COHERE_EMBED_FALLBACK_MODELS", "").strip()
COHERE_EMBED_FALLBACK_MODELS = [
    m.strip() for m in _fallback_raw.split(",") if m.strip()
]
COHERE_GENERATE_MODEL = os.getenv("COHERE_GENERATE_MODEL", "command-r-plus")

# Local options (T4-friendly defaults)
LOCAL_LLM_MODEL_ID = os.getenv("LOCAL_LLM_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
LOCAL_EMBED_MODEL_ID = os.getenv("LOCAL_EMBED_MODEL_ID", "BAAI/bge-m3")
LOCAL_DEVICE = os.getenv("LOCAL_DEVICE", "cuda")

# Prevent JAX from grabbing most GPU memory up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
