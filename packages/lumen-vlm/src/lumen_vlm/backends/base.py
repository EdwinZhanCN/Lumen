"""
Base backend interfaces for FastVLM-style multimodal generators.

This module defines runtime-agnostic abstractions that every FastVLM backend
(ONNX Runtime, custom accelerators, etc.) must implement so that higher layers
(Service, Model Manager, Registry) can remain identical across implementations.

Responsibilities covered here follow the Lumen development protocol:

1. Backend lifecycle management (initialize/close/is_initialized)
2. Prompt construction via chat templates (Jinja2-based) and fallbacks
3. Tokenizer loading utilities (Hugging Face `tokenizers` JSON format)
4. Definition of standard request/response dataclasses
5. Access to model metadata parsed from `model_info.json`

Concrete backends are expected to utilize `jinja2`, `tokenizers`, and
`onnxruntime` (or another runtime) to satisfy the abstract methods defined here.
"""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from jinja2 import Environment, StrictUndefined, TemplateError
from numpy.typing import NDArray
from tokenizers import Tokenizer

from ..resources.exceptions import ResourceNotFoundError
from .backend_exceptions import (
    BackendNotInitializedError,
    InvalidInputError,
    ModelLoadingError,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from lumen_vlm.resources.loader import ModelResources
else:  # pragma: no cover - runtime fallback until loader lands
    ModelResources = Any  # type: ignore[misc]


__all__ = [
    "ChatMessage",
    "GenerationConfig",
    "KVCacheConfig",
    "VisionConfig",
    "GenerationRequest",
    "GenerationChunk",
    "GenerationResult",
    "BackendInfo",
    "BaseFastVLMBackend",
]


@dataclass(frozen=True)
class ChatMessage:
    """Single chat message compatible with Hugging Face chat templates."""

    role: str
    content: str

    def to_mapping(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenerationConfig:
    """Token-level generation defaults extracted from `model_info.json`."""

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    image_token_index: int
    vocab_size: int
    max_position_embeddings: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationConfig":
        required = [
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "image_token_index",
            "vocab_size",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ModelLoadingError(
                f"generation_config missing required keys: {missing}"
            )
        return cls(
            bos_token_id=int(data["bos_token_id"]),
            eos_token_id=int(data["eos_token_id"]),
            pad_token_id=int(data["pad_token_id"]),
            image_token_index=int(data["image_token_index"]),
            vocab_size=int(data["vocab_size"]),
            max_position_embeddings=data.get("max_position_embeddings"),
        )


@dataclass(frozen=True)
class KVCacheConfig:
    """Static KV-cache metadata for decoder optimizations."""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KVCacheConfig":
        required = [
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "head_dim",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ModelLoadingError(f"kv_cache_config missing keys: {missing}")
        return cls(
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            hidden_size=int(data["hidden_size"]),
            head_dim=int(data["head_dim"]),
        )


@dataclass(frozen=True)
class VisionConfig:
    """Vision encoder preprocessing metadata."""

    image_size: int
    patch_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisionConfig":
        required = ["image_size", "patch_size", "mean", "std"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ModelLoadingError(f"vision_config missing keys: {missing}")

        def _vec(values: Sequence[float]) -> tuple[float, float, float]:
            if len(values) != 3:
                raise ModelLoadingError(
                    f"vision_config entries must have 3 values, got {values}"
                )
            return tuple(float(v) for v in values)  # type: ignore[return-value]

        return cls(
            image_size=int(data["image_size"]),
            patch_size=int(data["patch_size"]),
            mean=_vec(data["mean"]),
            std=_vec(data["std"]),
        )


@dataclass
class GenerationRequest:
    """Canonical generation request forwarded into backend implementers."""

    messages: Sequence[ChatMessage]
    image_bytes: bytes
    add_generation_prompt: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_sequences: Sequence[str] | None = None
    do_sample: bool = False
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationChunk:
    """Streaming chunk emitted by incremental decoders."""

    text: str
    tokens: list[int] = field(default_factory=list)
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Final aggregated generation result."""

    text: str
    tokens: list[int]
    finish_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendInfo:
    """Runtime + model metadata surfaced to service capability APIs."""

    runtime: str
    device: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    version: str | None = None
    precisions: list[str] = field(default_factory=list)
    max_new_tokens: int | None = None
    max_context_length: int | None = None
    vision_image_size: int | None = None
    vision_patch_size: int | None = None
    vocab_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        base: MutableMapping[str, Any] = {
            "runtime": self.runtime,
            "device": self.device,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "precisions": list(self.precisions),
            "max_new_tokens": self.max_new_tokens,
            "max_context_length": self.max_context_length,
            "vision_image_size": self.vision_image_size,
            "vision_patch_size": self.vision_patch_size,
            "vocab_size": self.vocab_size,
        }
        base.update(self.extra)
        return dict(base)


class BaseFastVLMBackend(abc.ABC):
    """Abstract base class for all FastVLM runtime backends."""

    def __init__(
        self,
        resources: ModelResources,
        device_preference: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.resources = resources
        self._device_preference = device_preference
        self._max_new_tokens = max_new_tokens
        self._initialized = False

        self._tokenizer: Tokenizer | None = None
        self._chat_template = self._extract_chat_template(resources)
        self._jinja_env = Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

        (
            self._generation_config,
            self._kv_cache_config,
            self._vision_config,
        ) = self._extract_metadata(resources)

        self._backend_info: BackendInfo | None = None

    # --------------------------------------------------------------------- #
    # Abstract API
    # --------------------------------------------------------------------- #

    @abc.abstractmethod
    def initialize(self) -> None:
        """Load weights and prepare runtime resources."""
        ...

    def close(self) -> None:
        """Release runtime resources (optional override)."""
        self._initialized = False

    @abc.abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult | Iterable[GenerationChunk]:
        """
        Run multimodal generation.

        Implementations may stream results (Iterable[GenerationChunk]) or return
        a single GenerationResult. Service layer will normalize either form.
        """

    @abc.abstractmethod
    def get_info(self) -> BackendInfo:
        """Return BackendInfo populated after initialize()."""

    # --------------------------------------------------------------------- #
    # Utility helpers shared by ONNX/Torch/etc. implementations
    # --------------------------------------------------------------------- #

    @property
    def device_preference(self) -> str | None:
        return self._device_preference

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @property
    def kv_cache_config(self) -> KVCacheConfig:
        return self._kv_cache_config

    @property
    def vision_config(self) -> VisionConfig:
        return self._vision_config

    @property
    def is_initialized(self) -> bool:
        """Check if the backend is properly initialized and ready for inference.

        Returns:
            bool: True if the backend has been successfully initialized with
                loaded models and configured devices, False otherwise.

        Note:
            This property should be checked before inference operations to avoid
            BackendNotInitializedError exceptions.
        """
        return self._initialized

    def ensure_initialized(self) -> None:
        if not self._initialized:
            raise BackendNotInitializedError(
                "Backend must be initialized before calling inference APIs."
            )

    # ----- Prompt utilities ------------------------------------------------ #

    def build_prompt(
        self,
        messages: Sequence[ChatMessage],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render chat template with fallback when template is absent/invalid."""
        if not messages:
            raise InvalidInputError("Chat messages cannot be empty.")

        template_str = self._chat_template
        if template_str:
            try:
                template = self._jinja_env.from_string(template_str)
                rendered = template.render(
                    messages=[m.to_mapping() for m in messages],
                    add_generation_prompt=add_generation_prompt,
                )
                if not isinstance(rendered, str):
                    raise TemplateError(
                        f"Template rendered non-string value ({type(rendered)})"
                    )
                return rendered.strip()
            except TemplateError as exc:
                logger.warning(
                    "Chat template rendering failed (%s). Falling back to naive prompt.",
                    exc,
                )

        # Fallback: simple role/content transcript similar to HF defaults
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<|{msg.role}|>\n{msg.content.strip()}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    # ----- Tokenizer utilities -------------------------------------------- #

    def tokenizer(self) -> Tokenizer:
        """Lazy-load Hugging Face Tokenizer from model cache."""
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizer_path = Path(self.resources.model_root_path) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise ResourceNotFoundError(
                f"Tokenizer file not found at {tokenizer_path}. "
                "FastVLM backends require tokenizer.json."
            )

        try:
            from tokenizers import Tokenizer as HFTokenizer  # local import

            self._tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModelLoadingError(
                "tokenizers package is required but not installed."
            ) from exc
        except Exception as exc:  # pragma: no cover
            raise ModelLoadingError(
                f"Failed to load tokenizer from {tokenizer_path}: {exc}"
            ) from exc

        if self._tokenizer is None:
            raise ModelLoadingError(f"Failed to load tokenizer from {tokenizer_path}")

        return self._tokenizer

    def tokenize(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        tokenizer = self.tokenizer()
        try:
            encoding = tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return encoding.ids  # type: ignore[return-value]
        except Exception as exc:
            raise InvalidInputError(f"Tokenization failed: {exc}") from exc

    def detokenize(self, token_ids: Sequence[int]) -> str:
        tokenizer = self.tokenizer()
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as exc:
            raise InvalidInputError(f"Detokenization failed: {exc}") from exc

    # ----- Request helpers ------------------------------------------------ #

    def build_generation_request(
        self,
        *,
        messages: Sequence[ChatMessage],
        image_bytes: bytes,
        **overrides: Any,
    ) -> GenerationRequest:
        """Merge overrides with defaults to produce a GenerationRequest."""
        if not image_bytes:
            raise InvalidInputError("Image payload is empty.")

        base_kwargs: dict[str, Any] = {
            "messages": messages,
            "image_bytes": image_bytes,
            "add_generation_prompt": overrides.pop(
                "add_generation_prompt",
                True,
            ),
            "max_new_tokens": overrides.pop(
                "max_new_tokens",
                self._max_new_tokens or 512,
            ),
            "temperature": overrides.pop("temperature", 0.0),
            "top_p": overrides.pop("top_p", 1.0),
            "repetition_penalty": overrides.pop("repetition_penalty", 1.0),
            "stop_sequences": overrides.pop("stop_sequences", None),
            "do_sample": overrides.pop("do_sample", False),
            "stream": overrides.pop("stream", False),
            "extra": overrides.pop("extra", {}),
        }

        if overrides:
            raise InvalidInputError(f"Unknown generation overrides: {list(overrides)}")

        return GenerationRequest(**base_kwargs)

    # ----- Metadata helpers ---------------------------------------------- #

    def _extract_metadata(
        self,
        resources: ModelResources,
    ) -> tuple[GenerationConfig, KVCacheConfig, VisionConfig]:
        meta = getattr(resources.model_info, "extra_metadata", None) or {}
        generation_cfg = GenerationConfig.from_dict(meta.get("generation_config", {}))
        kv_cfg = KVCacheConfig.from_dict(meta.get("kv_cache_config", {}))
        vision_cfg = VisionConfig.from_dict(meta.get("vision_config", {}))
        return generation_cfg, kv_cfg, vision_cfg

    def _extract_chat_template(self, resources: ModelResources) -> str | None:
        tokenizer_cfg = getattr(resources, "tokenizer_config", None)
        if isinstance(tokenizer_cfg, Mapping):
            template = tokenizer_cfg.get("chat_template")
            if isinstance(template, str) and template.strip():
                return template
        return None

    def _load_json(self, path: Path) -> Mapping[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError as exc:
            raise ResourceNotFoundError(f"Missing required file: {path}") from exc

    # ----- Backend info wiring ------------------------------------------- #

    def _set_backend_info(self, info: BackendInfo) -> None:
        self._backend_info = info
        self._initialized = True

    def _default_backend_info(self, runtime: str) -> BackendInfo:
        cfg = self.generation_config
        vision = self.vision_config
        return BackendInfo(
            runtime=runtime,
            device=self.device_preference,
            model_id=self.resources.model_info.name,
            model_name=self.resources.model_info.name,
            version=self.resources.model_info.version,
            precisions=["fp32", "fp16"],
            max_new_tokens=self._max_new_tokens,
            max_context_length=cfg.max_position_embeddings,
            vision_image_size=vision.image_size,
            vision_patch_size=vision.patch_size,
            vocab_size=cfg.vocab_size,
        )

    # --------------------------------------------------------------------- #
    # Convenience hooks for derived classes
    # --------------------------------------------------------------------- #

    def normalize_logits(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        """Utility for numerical stability when sampling."""
        logits = logits.astype(np.float32, copy=False)
        logits -= np.max(logits)
        return logits

    def stop_on_sequences(
        self,
        text: str,
        stop_sequences: Sequence[str] | None,
    ) -> tuple[str, str]:
        if not stop_sequences:
            return text, ""
        for stop in stop_sequences:
            idx = text.find(stop)
            if idx != -1:
                return text[:idx], stop
        return text, ""

    def build_stream_response(
        self,
        chunks: Iterable[GenerationChunk],
    ) -> Iterable[GenerationChunk]:
        """
        Hook for derived backends to massage low-level chunks.

        Default implementation just yields the provided iterator. Override if
        you need to enforce schema/metadata invariants.
        """
        return chunks
