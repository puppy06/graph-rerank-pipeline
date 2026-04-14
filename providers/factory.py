"""Provider factory based on config toggle."""

from __future__ import annotations

from config import (
    COHERE_API_KEY,
    COHERE_EMBED_API,
    COHERE_EMBED_FALLBACK_MODELS,
    COHERE_EMBED_MODEL,
    COHERE_EMBED_OUTPUT_DIMENSION,
    COHERE_GENERATE_MODEL,
    LOCAL_DEVICE,
    LOCAL_EMBED_MODEL_ID,
    LOCAL_LLM_MODEL_ID,
    USE_LOCAL_MODEL,
)
from providers.base import BaseModelProvider
from providers.cohere_client import CohereProvider
from providers.local_client import LocalProvider


def get_provider() -> BaseModelProvider:
    """
    Return exactly one backend: full Cohere or full local (Llama + BGE-M3).

    The two stacks are independent; only one is constructed per process.
    """
    if USE_LOCAL_MODEL:
        return LocalProvider(
            llm_model_id=LOCAL_LLM_MODEL_ID,
            embed_model_id=LOCAL_EMBED_MODEL_ID,
            device=LOCAL_DEVICE,
        )

    if not COHERE_API_KEY:
        raise ValueError(
            "COHERE_API_KEY is required when USE_LOCAL_MODEL=false."
        )

    return CohereProvider(
        api_key=COHERE_API_KEY,
        embed_model=COHERE_EMBED_MODEL,
        generate_model=COHERE_GENERATE_MODEL,
        embed_output_dimension=COHERE_EMBED_OUTPUT_DIMENSION,
        embed_api=COHERE_EMBED_API,
        embed_fallback_models=COHERE_EMBED_FALLBACK_MODELS,
    )
