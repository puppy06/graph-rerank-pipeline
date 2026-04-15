"""Cohere-backed provider implementation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from providers.base import BaseModelProvider

EmbedApiMode = Literal["v2", "v1", "auto"]


def _float_vectors_from_v2(response: Any) -> list[list[float]]:
    emb = response.embeddings
    floats = getattr(emb, "float", None)
    if floats is None:
        floats = getattr(emb, "float_", None)
    if floats is None:
        raise ValueError("Cohere v2 embed response missing float embeddings.")
    return floats


def _float_vectors_from_v1(response: Any) -> list[list[float]]:
    emb = response.embeddings
    if isinstance(emb, list):
        return emb
    floats = getattr(emb, "float", None)
    if floats is None:
        floats = getattr(emb, "float_", None)
    if floats is None:
        raise ValueError("Cohere v1 embed response missing float embeddings.")
    return floats


class CohereProvider(BaseModelProvider):
    """Provider that calls Cohere for both embeddings and generation."""

    def __init__(
        self,
        api_key: str,
        *,
        embed_model: str = "embed-v4.0",
        generate_model: str = "command-r-08-2024",
        embed_output_dimension: int | None = 1024,
        embed_api: EmbedApiMode | str = "auto",
        embed_fallback_models: list[str] | None = None,
    ) -> None:
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "cohere is required for CohereProvider. Install with `pip install cohere`."
            ) from exc

        from cohere.errors import NotFoundError

        self._NotFoundError = NotFoundError
        self._cohere = cohere

        self.embed_model = embed_model
        self.generate_model = generate_model
        self.embed_output_dimension = embed_output_dimension
        _api = str(embed_api).strip().lower()
        self.embed_api: EmbedApiMode = (
            _api if _api in ("v2", "v1", "auto") else "auto"
        )
        self.embed_fallback_models = list(embed_fallback_models or [])

        # v1 client: /v1/embed (often works when trial keys 404 on /v2/embed).
        self._client_v1: Any = cohere.Client(api_key)

        # Prefer V2 for chat and optional v2 embed.
        client_v2 = getattr(cohere, "ClientV2", None)
        if client_v2 is not None:
            self._client: Any = client_v2(api_key=api_key)
            self._use_v2 = True
        else:
            self._client = self._client_v1
            self._use_v2 = False

    def _embed_v2(
        self, model: str, texts: list[str], *, input_type: str
    ) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": model,
            "input_type": input_type,
            "texts": texts,
            "embedding_types": ["float"],
        }
        if model.startswith("embed-v4") and self.embed_output_dimension:
            kwargs["output_dimension"] = self.embed_output_dimension
        response = self._client.embed(**kwargs)
        return _float_vectors_from_v2(response)

    def _embed_v1(
        self, model: str, texts: list[str], *, input_type: str
    ) -> list[list[float]]:
        # batching=False avoids threaded batch path; try without embedding_types first
        # (some accounts route "light" models to packed variants only when types are set).
        try:
            response = self._client_v1.embed(
                texts=texts,
                model=model,
                input_type=input_type,
                batching=False,
            )
            return _float_vectors_from_v1(response)
        except self._NotFoundError:
            raise
        except Exception:
            response = self._client_v1.embed(
                texts=texts,
                model=model,
                input_type=input_type,
                embedding_types=["float"],
                batching=False,
            )
            return _float_vectors_from_v1(response)

    def _embed_for_input_type(self, texts: list[str], *, input_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if not self._use_v2:
            vectors = self._embed_v1(
                self.embed_model, texts, input_type=input_type
            )
            return np.asarray(vectors, dtype=np.float32)

        models_to_try = [self.embed_model, *self.embed_fallback_models]
        seen: set[str] = set()
        last_err: Exception | None = None

        for model in models_to_try:
            if model in seen:
                continue
            seen.add(model)

            if self.embed_api in ("v2", "auto"):
                try:
                    vectors = self._embed_v2(model, texts, input_type=input_type)
                    return np.asarray(vectors, dtype=np.float32)
                except self._NotFoundError as e:
                    last_err = e
                    if self.embed_api == "v2":
                        continue
                except Exception:
                    raise

            if self.embed_api in ("v1", "auto"):
                try:
                    vectors = self._embed_v1(model, texts, input_type=input_type)
                    return np.asarray(vectors, dtype=np.float32)
                except self._NotFoundError as e:
                    last_err = e
                except Exception:
                    raise

        msg = (
            "Cohere embed failed for all models tried. "
            "Use a Cohere key with Embed enabled, or set USE_LOCAL_MODEL=true "
            "to use the full local stack (BGE-M3 + Llama) instead. "
            f"Tried (in order): {', '.join(dict.fromkeys(models_to_try))}. "
            "See https://dashboard.cohere.com/"
        )
        if last_err is not None:
            raise RuntimeError(msg) from last_err
        raise RuntimeError(msg)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed_for_input_type(texts, input_type="search_document")

    def embed_query(self, texts: list[str]) -> np.ndarray:
        return self._embed_for_input_type(texts, input_type="search_query")

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        if self._use_v2:
            response = self._client.chat(
                model=self.generate_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
            return response.message.content[0].text.strip()

        response = self._client.chat(
            model=self.generate_model,
            message=prompt,
            max_tokens=max_new_tokens,
        )
        return response.text.strip()
