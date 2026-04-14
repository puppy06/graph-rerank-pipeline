"""Local (GPU) provider using Hugging Face Transformers + bitsandbytes."""

from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from providers.base import BaseModelProvider

# Keep JAX from pre-allocating almost all GPU memory so local LLM + JAX can coexist.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


class LocalProvider(BaseModelProvider):
    """
    Local provider for:
    - generation: Llama 3 (4-bit quantized via bitsandbytes)
    - embeddings: BGE-M3
    """

    def __init__(
        self,
        *,
        llm_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        embed_model_id: str = "BAAI/bge-m3",
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.llm_model_id = llm_model_id
        self.embed_model_id = embed_model_id

        self._llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id)
        if self._llm_tokenizer.pad_token is None:
            self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_id,
            device_map="auto" if device == "cuda" else None,
            quantization_config=quant_config if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        self._embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_id)
        self._embed_model = AutoModel.from_pretrained(self.embed_model_id)
        self._embed_model.to(self.device)
        self._embed_model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        with torch.no_grad():
            batch = self._embed_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(self.device)
            outputs = self._embed_model(**batch)
            pooled = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return normalized.detach().cpu().numpy().astype(np.float32)

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = self._llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._llm_tokenizer(
            rendered_prompt,
            return_tensors="pt",
        ).to(self._llm.device)

        with torch.no_grad():
            output_ids = self._llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=self._llm_tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        return self._llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
