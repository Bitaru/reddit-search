"""Local Qwen embedding and reranking adapters.

The adapters are intentionally offline-only: model revisions are required and
Hugging Face loading uses ``local_files_only=True``. Provisioning model files is
an explicit operator step, never an implicit runtime download.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .dense import EmbeddingAdapter, EmbeddingRecipe
from .rerank import RerankerAdapter

_MAX_SEQUENCE_LENGTH = 8_192
_RERANK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised by preflight
        raise RuntimeError(
            "local Qwen adapters require torch and transformers; "
            "install the semantic dependency group"
        ) from error
    return torch, (AutoModel, AutoModelForCausalLM), AutoTokenizer


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported local model device: {requested}")
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _load_kwargs(
    *, revision: str, cache_dir: str | Path | None, torch: Any, device: Any
) -> dict[str, Any]:
    if not revision.strip():
        raise ValueError("model revision must be pinned")
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": True,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if device.type in {"mps", "cuda"}:
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32
    return kwargs


class QwenEmbeddingAdapter(EmbeddingAdapter):
    """Encode Qwen documents and prompted queries from a local model cache."""

    def __init__(
        self,
        recipe: EmbeddingRecipe,
        *,
        device: str = "auto",
        batch_size: int = 8,
        cache_dir: str | Path | None = None,
        max_length: int = _MAX_SEQUENCE_LENGTH,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.recipe = recipe
        self.batch_size = batch_size
        self.max_length = max_length
        torch, model_classes, tokenizer_class = _load_runtime()
        auto_model, _ = model_classes
        self._torch = torch
        self._device = _resolve_device(torch, device)
        load_kwargs = _load_kwargs(
            revision=recipe.revision,
            cache_dir=cache_dir,
            torch=torch,
            device=self._device,
        )
        self._tokenizer = tokenizer_class.from_pretrained(
            recipe.model_id,
            padding_side="left",
            **load_kwargs,
        )
        self._model = auto_model.from_pretrained(recipe.model_id, **load_kwargs)
        self._model.to(self._device)
        self._model.eval()

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._encode(texts, prompted=False)

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._encode(texts, prompted=True)

    def _encode(self, texts: Sequence[str], *, prompted: bool) -> list[list[float]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embedding inputs must be strings")
        if not texts:
            return []
        if prompted:
            inputs = [f"Instruct: {self.recipe.query_instruction}\nQuery:{text}" for text in texts]
        else:
            inputs = list(texts)
        ordered = sorted(enumerate(inputs), key=lambda item: len(item[1]))
        vectors: list[list[float] | None] = [None] * len(inputs)
        for start in range(0, len(ordered), self.batch_size):
            batch_items = ordered[start : start + self.batch_size]
            encoded = self._tokenizer(
                [text for _, text in batch_items],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.inference_mode():
                outputs = self._model(**encoded).last_hidden_state
                attention_mask = encoded["attention_mask"]
                positions = attention_mask.sum(dim=1) - 1
                batch_positions = self._torch.arange(outputs.shape[0], device=outputs.device)
                pooled = outputs[batch_positions, positions]
                pooled = self._torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                if self._device.type == "mps":
                    self._torch.mps.synchronize()
            if pooled.shape[1] != self.recipe.dimension:
                raise RuntimeError(
                    f"embedding dimension mismatch: expected {self.recipe.dimension}, "
                    f"got {pooled.shape[1]}"
                )
            cpu_vectors = pooled.detach().to("cpu").tolist()
            for (original_index, _), vector in zip(
                batch_items,
                cpu_vectors,
                strict=True,
            ):
                vectors[original_index] = vector
            del cpu_vectors, pooled, outputs, attention_mask, batch_positions, encoded
        return [vector for vector in vectors if vector is not None]


class QwenRerankerAdapter(RerankerAdapter):
    """Score query/document pairs with the local Qwen yes/no reranker head."""

    model_id: str
    revision: str

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        cache_dir: str | Path | None = None,
        max_length: int = _MAX_SEQUENCE_LENGTH,
    ) -> None:
        if not model_id.strip():
            raise ValueError("reranker model_id must not be empty")
        if not revision.strip():
            raise ValueError("reranker revision must be pinned")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        torch, model_classes, tokenizer_class = _load_runtime()
        _, causal_model = model_classes
        self._torch = torch
        self._device = _resolve_device(torch, device)
        load_kwargs = _load_kwargs(
            revision=revision,
            cache_dir=cache_dir,
            torch=torch,
            device=self._device,
        )
        self._tokenizer = tokenizer_class.from_pretrained(
            model_id,
            padding_side="left",
            **load_kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = causal_model.from_pretrained(model_id, **load_kwargs)
        self._model.to(self._device)
        self._model.eval()
        self._true_token_id = self._single_token_id("yes")
        self._false_token_id = self._single_token_id("no")

    def score(
        self,
        *,
        scenario_query: str,
        candidate_texts: Sequence[str],
        instruction: str,
    ) -> Sequence[float]:
        if not scenario_query.strip() or not instruction.strip():
            raise ValueError("reranker query and instruction must not be empty")
        if any(not isinstance(text, str) for text in candidate_texts):
            raise TypeError("reranker candidate texts must be strings")
        if not candidate_texts:
            return []
        encoded_pairs = [
            self._format_pair(instruction, scenario_query, text) for text in candidate_texts
        ]
        scores: list[float] = []
        for start in range(0, len(encoded_pairs), self.batch_size):
            batch = encoded_pairs[start : start + self.batch_size]
            inputs = self._tokenizer.pad(
                [{"input_ids": input_ids} for input_ids in batch],
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.inference_mode():
                logits = self._model(**inputs).logits[:, -1, :]
                selected = logits[:, [self._false_token_id, self._true_token_id]]
                batch_scores = self._torch.softmax(selected, dim=1)[:, 1]
            if self._device.type == "mps":
                self._torch.mps.synchronize()
            scores.extend(float(score) for score in batch_scores.detach().to("cpu"))
            del batch_scores, selected, logits, inputs
        return scores

    def _single_token_id(self, text: str) -> int:
        token_ids = self._tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise RuntimeError(f"reranker token {text!r} is not represented by one token")
        return int(token_ids[0])

    def _format_pair(self, instruction: str, query: str, document: str) -> list[int]:
        prefix_tokens = self._tokenizer.encode(_RERANK_PREFIX, add_special_tokens=False)
        suffix_tokens = self._tokenizer.encode(_RERANK_SUFFIX, add_special_tokens=False)
        body = f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"
        body_tokens = self._tokenizer.encode(body, add_special_tokens=False)
        available = self.max_length - len(prefix_tokens) - len(suffix_tokens)
        if available <= 0:
            raise ValueError("reranker max_length is too small for its prompt template")
        return prefix_tokens + body_tokens[:available] + suffix_tokens
