"""
ONNX Runtime backend implementation for FastVLM.

This backend orchestrates the ONNX runtimes for the three FastVLM sub-models
(vision encoder, text embedder, causal decoder) and exposes the unified
`BaseFastVLMBackend` interface.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from typing_extensions import override

from ..resources.exceptions import ResourceNotFoundError
from .backend_exceptions import (
    BackendNotInitializedError,
    InvalidInputError,
    ModelLoadingError,
)
from .base import (
    BackendInfo,
    BaseFastVLMBackend,
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
)

logger = logging.getLogger(__name__)

_has_ort = importlib.util.find_spec("onnxruntime") is not None

if TYPE_CHECKING:
    import onnxruntime as ort
else:
    if _has_ort:
        import onnxruntime as ort
    else:
        ort = None


class ONNXRuntimeNotAvailableError(ModelLoadingError):
    """Raised when onnxruntime cannot be imported."""


class FastVLMONNXBackend(BaseFastVLMBackend):
    """ONNX Runtime backend for FastVLM models."""

    # LLaVA/Qwen2-VL specific: Image placeholder token ID
    IMAGE_TOKEN_ID = 151646

    def __init__(
        self,
        resources,
        *,
        providers: list[str] | None = None,
        device_preference: str | None = None,
        max_new_tokens: int | None = None,
        prefer_fp16: bool = True,
    ) -> None:
        super().__init__(
            resources=resources,
            device_preference=device_preference,
            max_new_tokens=max_new_tokens,
        )
        self._providers = providers or self._default_providers(device_preference)
        self._prefer_fp16 = prefer_fp16

        # Sessions
        self._sess_vision: ort.InferenceSession | None = None
        self._sess_embed: ort.InferenceSession | None = None
        self._sess_decoder: ort.InferenceSession | None = None

        # IO Metadata
        self._vision_input_name: str | None = None
        self._vision_output_name: str | None = None

        # Embedder is now text-only
        self._embed_input_name: str | None = None
        self._embed_output_name: str | None = None

        # Decoder inputs (complex)
        self._decoder_input_names: list[str] = []
        self._decoder_output_names: list[str] = []

        # Data type tracking for decoder
        self._decoder_embed_dtype: np.dtype | None = None
        self._decoder_kv_dtype: np.dtype | None = None
        self._decoder_input_dtypes: dict[str, np.dtype] = {}

        self._vision_dtype = np.float32
        # self._vision_image_size is loaded from config via base class property
        self._load_time_seconds: float | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @override
    def initialize(self) -> None:
        if self._initialized:
            return

        t0 = time.time()
        runtime_root = Path(self.resources.model_root_path) / "onnx"
        logger.info(f"Providers: {self._providers}")
        try:
            self._sess_vision = self._build_session(
                runtime_root, "vision", prefer_fp16=self._prefer_fp16
            )
            self._sess_embed = self._build_session(runtime_root, "embed")
            self._sess_decoder = self._build_session(runtime_root, "decoder")

            self._wire_io_metadata()
            backend_info = self._default_backend_info(runtime="onnxruntime")
            backend_info.precisions = self._detect_precisions()
            backend_info.extra.update(
                {
                    "providers": ",".join(self._providers),
                    "load_time_s": f"{time.time() - t0:.2f}",
                }
            )
            self._set_backend_info(backend_info)
            self._load_time_seconds = time.time() - t0
            self._initialized = True
            logger.info(
                "✅ FastVLM ONNX backend initialized in %.2fs", self._load_time_seconds
            )
        except Exception as exc:
            raise ModelLoadingError(
                f"Failed to initialize ONNX backend: {exc}"
            ) from exc

    def close(self) -> None:
        self._sess_vision = None
        self._sess_embed = None
        self._sess_decoder = None
        self._backend_info = None
        self._initialized = False

    @override
    def get_info(self) -> "BackendInfo":
        if self._backend_info is None:
            raise BackendNotInitializedError(
                "Backend must be initialized before getting info. Call initialize() first."
            )
        return self._backend_info

    # ------------------------------------------------------------------ #
    # Generation (Refactored for LLaVA Logic)
    # ------------------------------------------------------------------ #
    @override
    def generate(
        self, request: GenerationRequest
    ) -> GenerationResult | Iterable[GenerationChunk]:
        self.ensure_initialized()

        # 1. Prepare Inputs
        prompt = self.build_prompt(
            messages=request.messages,
            add_generation_prompt=request.add_generation_prompt,
        )

        # 添加调试日志
        logger.info(f"Built prompt: {prompt}")
        logger.info(f"Messages: {request.messages}")

        text_tokens = self.tokenize(prompt)
        logger.info(f"Tokenized prompt: {text_tokens}")

        # 2. Vision Encoding
        logger.info(f"Image bytes size: {len(request.image_bytes)}")
        vision_embeds = self._run_vision_encoder(request.image_bytes)

        # 3. Text Embedding
        # Output: [1, seq_len, hidden_size]
        text_embeds = self._run_text_embedding(text_tokens)

        # 4. Stitching (Merge)
        inputs_embeds, attention_mask = self._merge_embeddings(
            text_tokens, text_embeds, vision_embeds
        )

        # 5. Initialize KV Cache
        kv_cache = self._prepare_kv_cache()

        # 6. Prefill Phase (Run Decoder on full sequence)
        seq_len = inputs_embeds.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

        # Convert merged embeddings to decoder's expected dtype if needed
        if (
            self._decoder_embed_dtype is not None
            and inputs_embeds.dtype != self._decoder_embed_dtype
        ):
            logger.info(
                f"Converting merged embeddings from {inputs_embeds.dtype} to {self._decoder_embed_dtype}"
            )
            inputs_embeds = inputs_embeds.astype(self._decoder_embed_dtype)

        logits, kv_cache = self._run_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # Sample first token
        next_token = self._sample(logits, request.temperature, request.top_p)

        # 7. Decode Phase
        if request.stream:
            chunks = self._stream_generate(
                first_token=next_token,
                past_seq_len=seq_len,
                kv_cache=kv_cache,
                request=request,
            )
            return self.build_stream_response(chunks)
        else:
            return self._full_generate(
                first_token=next_token,
                past_seq_len=seq_len,
                kv_cache=kv_cache,
                request=request,
            )

    # ------------------------------------------------------------------ #
    # Core Logic Implementation
    # ------------------------------------------------------------------ #

    def _merge_embeddings(
        self,
        input_ids: list[int],
        text_embeds: NDArray[np.float32],
        vision_embeds: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Splits text embeddings at <image> token and inserts vision embeddings."""
        logger.info(f"Input IDs length: {len(input_ids)}")
        logger.info(f"Input IDs: {input_ids}")
        logger.info(f"Text embeddings shape: {text_embeds.shape}")
        logger.info(f"Vision embeddings shape: {vision_embeds.shape}")

        # 检查是否有图像标记
        if self.IMAGE_TOKEN_ID not in input_ids:
            logger.warning("No <image> token found in input_ids! Using text-only mode.")
            seq_len = text_embeds.shape[1]
            return text_embeds, np.ones((1, seq_len), dtype=np.int64)

        image_idx = input_ids.index(self.IMAGE_TOKEN_ID)
        logger.info(f"Found <image> token at position {image_idx}")

        # 检查维度
        if vision_embeds.shape[0] != text_embeds.shape[0]:
            logger.error(
                f"Batch size mismatch: text={text_embeds.shape[0]}, vision={vision_embeds.shape[0]}"
            )
            raise ValueError("Batch size mismatch between text and vision embeddings")

        if vision_embeds.shape[2] != text_embeds.shape[2]:
            logger.error(
                f"Hidden size mismatch: text={text_embeds.shape[2]}, vision={vision_embeds.shape[2]}"
            )
            logger.warning("Trying to adjust vision embeddings dimensions...")
            # 如果维度不匹配，尝试调整
            if vision_embeds.shape[2] == 3072 and text_embeds.shape[2] == 896:
                logger.warning(
                    "Detected typical dimension mismatch: vision(3072) -> text(896)"
                )
                # 可能需要一个投影层，这里先使用平均池化作为临时解决方案
                vision_embeds = vision_embeds.mean(axis=2, keepdims=True)
                vision_embeds = np.repeat(vision_embeds, 896, axis=2)

        # 分割文本嵌入
        part_a = text_embeds[:, :image_idx, :]
        part_b = text_embeds[:, image_idx + 1 :, :]

        logger.info(f"Text part A shape: {part_a.shape}")
        logger.info(f"Text part B shape: {part_b.shape}")

        # 合并嵌入
        merged = np.concatenate([part_a, vision_embeds, part_b], axis=1)
        logger.info(f"Merged embeddings shape: {merged.shape}")

        # 创建注意力掩码
        mask = np.ones((1, merged.shape[1]), dtype=np.int64)

        return merged, mask

    def _full_generate(
        self,
        first_token: int,
        past_seq_len: int,
        kv_cache: dict[str, NDArray[np.float32]],
        request: GenerationRequest,
    ) -> GenerationResult:
        logger.info(
            f"Starting full generation with max_new_tokens={request.max_new_tokens}, "
            f"temperature={request.temperature}, top_p={request.top_p}"
        )

        generated_tokens = [first_token]
        current_token = first_token
        current_seq_len = past_seq_len

        for step in range(request.max_new_tokens - 1):
            # Embed single token
            token_embed = self._run_text_embedding([current_token])

            # Prepare single-step inputs
            pos_ids = np.array([[current_seq_len]], dtype=np.int64)
            mask = np.ones((1, current_seq_len + 1), dtype=np.int64)

            logits, kv_cache = self._run_decoder(
                inputs_embeds=token_embed,
                attention_mask=mask,
                position_ids=pos_ids,
                kv_cache=kv_cache,
            )

            next_token = self._sample(logits, request.temperature, request.top_p)

            if next_token == self.generation_config.eos_token_id:
                logger.info(
                    f"Generation stopped at EOS token after {len(generated_tokens)} tokens"
                )
                break

            generated_tokens.append(next_token)
            current_token = next_token
            current_seq_len += 1

        text = self.detokenize(generated_tokens)

        truncated, stop = self.stop_on_sequences(text, request.stop_sequences)

        logger.info(
            f"Generation completed: {len(generated_tokens)} tokens, "
            f"finish_reason={'stop' if stop else 'length'}, "
            f"output_length={len(truncated)} chars"
        )

        return GenerationResult(
            text=truncated,
            tokens=generated_tokens,
            finish_reason="stop" if stop else "length",
            metadata={"tokens_generated": len(generated_tokens)},
        )

    def _stream_generate(
        self,
        first_token: int,
        past_seq_len: int,
        kv_cache: dict[str, NDArray[np.float32]],
        request: GenerationRequest,
    ) -> Generator[GenerationChunk, None, None]:
        current_token = first_token
        current_seq_len = past_seq_len
        generated_tokens = [current_token]
        buffer = self.detokenize([current_token])

        yield GenerationChunk(
            text=buffer, tokens=[current_token], is_final=False, metadata={"step": 0}
        )

        for step in range(request.max_new_tokens - 1):
            # Embed single token
            token_embed = self._run_text_embedding([current_token])

            # Prepare inputs
            pos_ids = np.array([[current_seq_len]], dtype=np.int64)
            mask = np.ones((1, current_seq_len + 1), dtype=np.int64)

            logits, kv_cache = self._run_decoder(
                inputs_embeds=token_embed,
                attention_mask=mask,
                position_ids=pos_ids,
                kv_cache=kv_cache,
            )

            next_token = self._sample(logits, request.temperature, request.top_p)

            if next_token == self.generation_config.eos_token_id:
                break

            generated_tokens.append(next_token)
            new_text = self.detokenize([next_token])
            buffer += new_text

            truncated, stop = self.stop_on_sequences(buffer, request.stop_sequences)
            if stop:
                # Yield remaining if needed, then stop
                yield GenerationChunk(
                    text="", tokens=[], is_final=True, metadata={"reason": "stop_seq"}
                )
                return

            yield GenerationChunk(
                text=new_text,
                tokens=[next_token],
                is_final=False,
                metadata={"step": step + 1},
            )

            current_token = next_token
            current_seq_len += 1

        yield GenerationChunk(
            text="", tokens=[], is_final=True, metadata={"reason": "max_length"}
        )

    def _run_decoder(
        self,
        inputs_embeds: NDArray[np.float32],
        attention_mask: NDArray[np.int64],
        position_ids: NDArray[np.int64],
        kv_cache: dict[str, NDArray[np.float32]],
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        """Runs the decoder and manages KV cache rotation."""

        # 1. Map Inputs with type conversion
        # We try to match standard names, falling back to discovered names if needed
        inputs = {}

        for name in self._decoder_input_names:
            if "embed" in name:
                # Convert embeddings to the expected dtype
                target_dtype = (
                    self._decoder_embed_dtype
                    if self._decoder_embed_dtype is not None
                    else np.float32
                )
                if inputs_embeds.dtype != target_dtype:
                    logger.info(
                        f"Converting embeddings from {inputs_embeds.dtype} to {target_dtype}"
                    )
                    inputs[name] = inputs_embeds.astype(target_dtype)
                else:
                    inputs[name] = inputs_embeds

            elif "mask" in name:
                # Attention mask typically stays int64
                inputs[name] = attention_mask

            elif "pos" in name:
                # Position IDs typically stay int64
                inputs[name] = position_ids

        # Add KV cache inputs with type validation
        for cache_name, cache_tensor in kv_cache.items():
            if cache_name in self._decoder_input_names:
                expected_dtype = self._decoder_input_dtypes.get(cache_name)
                if expected_dtype is not None and cache_tensor.dtype != expected_dtype:
                    logger.warning(
                        f"KV cache tensor '{cache_name}' dtype mismatch: got {cache_tensor.dtype}, expected {expected_dtype}"
                    )
                    # Convert to expected dtype
                    inputs[cache_name] = cache_tensor.astype(expected_dtype)
                    logger.info(
                        f"Converted KV cache '{cache_name}' from {cache_tensor.dtype} to {expected_dtype}"
                    )
                else:
                    inputs[cache_name] = cache_tensor

        # 2. Run inference
        if self._sess_decoder is None:
            raise RuntimeError("Decoder session is not initialized")
        outputs = self._sess_decoder.run(self._decoder_output_names, inputs)

        # 3. Process Outputs (Logits + KV)
        logits = cast(NDArray[np.float32], outputs[0])

        new_kv = {}
        # Map 'present' outputs back to 'past' inputs for next step
        for i, out_name in enumerate(self._decoder_output_names[1:]):
            # Heuristic: replace "present" with "past_key_values"
            # Adjust this logic if your model exports names differently (e.g. just "past")
            key_in = out_name.replace("present", "past_key_values")
            if "key_values" not in key_in:
                key_in = out_name.replace("present", "past")

            new_kv[key_in] = outputs[i + 1]

        return logits, new_kv

    def _run_text_embedding(
        self, input_ids: list[int] | Sequence[int]
    ) -> NDArray[np.float32]:
        ids = np.array([input_ids], dtype=np.int64)

        if self._sess_embed is None:
            raise RuntimeError("Embed session is not initialized")

        out = self._sess_embed.run(
            [self._embed_output_name], {self._embed_input_name: ids}
        )
        text_embeds = cast(NDArray[np.float32], out[0])
        return text_embeds

    def _sample(self, logits: NDArray[np.float32], temp: float, top_p: float) -> int:
        next_token_logits = logits[0, -1, :]
        if temp < 1e-5:
            return int(np.argmax(next_token_logits))

        # Temp Scaling
        next_token_logits = next_token_logits / temp
        probs = self._softmax(next_token_logits)

        # Top-P (Simple implementation)
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices_to_remove[np.argsort(sorted_indices)]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    # ------------------------------------------------------------------ #
    # Internal helpers (Preserved / Adjusted)
    # ------------------------------------------------------------------ #
    def _build_session(
        self,
        runtime_root: Path,
        component: str,
        *,
        prefer_fp16: bool = True,
    ) -> ort.InferenceSession:
        if ort is None:
            raise ONNXRuntimeNotAvailableError("onnxruntime not available")

        model_path = self._select_model_file(runtime_root, component, prefer_fp16)
        logger.info("Loading %s model from %s", component, model_path)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        return ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=self._providers,
        )

    def _select_model_file(
        self,
        runtime_root: Path,
        component: str,
        prefer_fp16: bool,
    ) -> Path:
        # Adjusted prefix mapping for your specific file names
        prefix_map = {
            "vision": "vision",
            "embed": "embed",
            "decoder": "decoder",
        }
        # Fallback to simple names if needed, or keep strictly to what you have
        if component not in prefix_map:
            prefix_map[component] = component

        prefix = prefix_map[component]
        # Common naming patterns
        candidates = []
        if prefer_fp16:
            candidates.append(runtime_root / f"{prefix}.fp16.onnx")
            candidates.append(runtime_root / f"{prefix}_fp16.onnx")

        candidates.append(runtime_root / f"{prefix}.onnx")
        candidates.append(runtime_root / f"{prefix}.fp32.onnx")

        for path in candidates:
            if path.exists():
                return path

        raise ResourceNotFoundError(
            f"Missing ONNX file for component '{component}' (checked {candidates})"
        )

    def _wire_io_metadata(self) -> None:
        assert self._sess_vision and self._sess_embed and self._sess_decoder

        # Vision
        vis_input = self._sess_vision.get_inputs()[0]
        self._vision_input_name = vis_input.name
        self._vision_output_name = self._sess_vision.get_outputs()[0].name
        self._vision_dtype = self._onnx_type_to_dtype(vis_input.type)
        logger.info(f"Vision encoder input dtype: {self._vision_dtype}")

        # Embed (Text only now)
        embed_inputs = self._sess_embed.get_inputs()
        self._embed_input_name = embed_inputs[0].name
        self._embed_output_name = self._sess_embed.get_outputs()[0].name

        # Decoder (Capture all)
        self._decoder_input_names = [
            inp.name for inp in self._sess_decoder.get_inputs()
        ]
        self._decoder_output_names = [
            out.name for out in self._sess_decoder.get_outputs()
        ]

        # Enhanced: Detect and store decoder data types
        logger.info("Detecting decoder input data types...")
        for inp in self._sess_decoder.get_inputs():
            dtype = self._onnx_type_to_dtype(inp.type)
            self._decoder_input_dtypes[inp.name] = dtype

            # Identify embedding input (typically contains 'embed' or is the first input)
            if "embed" in inp.name.lower() or (
                len(self._decoder_input_names) > 0
                and inp.name == self._decoder_input_names[0]
                and "key" not in inp.name.lower()
                and "value" not in inp.name.lower()
                and "mask" not in inp.name.lower()
                and "pos" not in inp.name.lower()
            ):
                self._decoder_embed_dtype = dtype
                logger.info(f"Decoder embedding input '{inp.name}' dtype: {dtype}")

            # Identify KV cache inputs (contain 'key' or 'value')
            elif "key" in inp.name.lower() or "value" in inp.name.lower():
                if self._decoder_kv_dtype is None:
                    self._decoder_kv_dtype = dtype
                    logger.info(
                        f"Decoder KV cache dtype detected: {dtype} from input '{inp.name}'"
                    )

        # Fallback if embed dtype not detected
        if self._decoder_embed_dtype is None:
            # Assume first non-KV, non-mask, non-pos input is embedding
            for name in self._decoder_input_names:
                if not any(
                    k in name.lower()
                    for k in ["key", "value", "mask", "position", "pos"]
                ):
                    self._decoder_embed_dtype = self._decoder_input_dtypes[name]
                    logger.info(
                        f"Decoder embedding dtype fallback: {self._decoder_embed_dtype} from input '{name}'"
                    )
                    break

        logger.info(
            f"Decoder data types - Embeddings: {self._decoder_embed_dtype}, KV Cache: {self._decoder_kv_dtype}"
        )

    def _run_vision_encoder(self, image_bytes: bytes) -> NDArray[np.float32]:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise InvalidInputError(f"Failed to decode image: {exc}") from exc

        # 根据 config.json 中的 image_aspect_ratio 和 preprocessor_config.json 的配置
        image_aspect = getattr(self.vision_config, "image_aspect_ratio", "pad")
        if image_aspect == "pad":
            # 实现 padding 逻辑
            target_width = 1024
            target_height = 1024

            # 计算缩放比例
            width, height = image.size
            scale = min(target_width / width, target_height / height)
            new_w = int(width * scale)
            new_h = int(height * scale)

            # 缩放图像
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

            # 创建新图像并居中放置
            new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            paste_x = (target_width - new_w) // 2
            paste_y = (target_height - new_h) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        else:
            # 如果配置要求裁剪，则进行中心裁剪
            if getattr(self.vision_config, "do_center_crop", True):
                # 进行中心裁剪
                width, height = image.size
                target_width = 1024
                target_height = 1024

                # 计算裁剪区域
                left = (width - target_width) // 2
                top = (height - target_height) // 2
                right = left + target_width
                bottom = top + target_height

                image = image.crop((left, top, right, bottom))
            else:
                # 直接调整大小
                image = image.resize((1024, 1024), Image.Resampling.BICUBIC)

        # 转换为 numpy 数组并进行归一化
        arr = np.array(image, dtype=np.float32) / 255.0  # 转换为 [0, 1] 范围

        # 应用均值归一化
        mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        arr = (arr - mean) / std

        # 转换为 NCHW 格式 [1, 3, 1024, 1024]
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0).astype(self._vision_dtype)

        assert self._sess_vision and self._vision_input_name
        outputs = self._sess_vision.run(
            [self._vision_output_name],
            {self._vision_input_name: arr},
        )

        vision_embeds = np.asarray(outputs[0], dtype=np.float32)

        return vision_embeds

    def _prepare_kv_cache(self) -> dict[str, NDArray[np.float32]]:
        cfg = self.kv_cache_config

        # Use detected KV cache dtype or default to float32
        kv_dtype = (
            self._decoder_kv_dtype if self._decoder_kv_dtype is not None else np.float32
        )
        logger.info(f"Preparing KV cache with dtype: {kv_dtype}")

        cache: dict[str, NDArray[np.float32]] = {}
        for layer in range(cfg.num_hidden_layers):
            cache[f"past_key_values.{layer}.key"] = np.zeros(
                (cfg.num_key_value_heads, 0, cfg.head_dim),
                dtype=kv_dtype,
            )
            cache[f"past_key_values.{layer}.value"] = np.zeros(
                (cfg.num_key_value_heads, 0, cfg.head_dim),
                dtype=kv_dtype,
            )

        # Ensure batch dim is 1 (Standard ONNX export: [batch, num_heads, seq, head_dim])
        for k in cache:
            cache[k] = np.expand_dims(cache[k], axis=0)

        return cache

    def _detect_precisions(self) -> list[str]:
        precisions: list[str] = []

        def _label(sess: ort.InferenceSession | None) -> str | None:
            if not sess:
                return None
            try:
                first_input = sess.get_inputs()[0]
                dtype = self._onnx_type_to_dtype(first_input.type)
                return "fp16" if dtype == np.float16 else "fp32"
            except Exception:
                return "fp32"

        for sess in (self._sess_vision, self._sess_embed, self._sess_decoder):
            label = _label(sess)
            if label and label not in precisions:
                precisions.append(label)
        return precisions or ["fp32"]

    def _default_providers(self, device_pref: str | None) -> list[str]:
        available = set(ort.get_available_providers())
        priority = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected = [prov for prov in priority if prov in available]

        pref_map = {
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
            "directml": "DmlExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
        }
        desired = pref_map.get((device_pref or "").lower())
        if desired and desired in selected:
            selected.insert(0, selected.pop(selected.index(desired)))

        return selected or ["CPUExecutionProvider"]

    @staticmethod
    def _onnx_type_to_dtype(type_str: str) -> Any:
        if "float16" in type_str.lower():
            return np.float16
        if "int64" in type_str.lower():
            return np.int64
        return np.float32

    @staticmethod
    def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
        exps = np.exp(logits - np.max(logits))
        return exps / np.sum(exps)
