"""
unified_service.py

A unified, high-performance gRPC service that combines the capabilities of
general CLIP and BioCLIP models.

Features:
- Exposes 7 tasks for classification and embedding (including image/text embeddings).
- Implements server-side request batching to maximize throughput.
- Uses backend-driven image batch embedding via backend.image_batch_to_vectors()
  (each backend may fall back to sequential processing if true batching is unavailable).
- No torchvision-based preprocessing here; image decoding/preprocessing is handled inside backends.
"""
import json
import logging
import os
import time
from collections import defaultdict
from typing import Iterable, List, Tuple

import grpc
import torch



from google.protobuf import empty_pb2
from backends import TorchBackend, ONNXRTBackend, RKNNBackend

# Import the service definition and model managers
import ml_service_pb2 as pb
import ml_service_pb2_grpc as rpc
from biological_atlas import BioCLIPModelManager
from image_classification import CLIPModelManager

# --- Constants ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
logger = logging.getLogger(__name__)







class CLIPService(rpc.InferenceServicer):
    """A single gRPC service that intelligently routes requests to CLIP or BioCLIP models."""

    SERVICE_NAME = "unified-ml-service"

    def __init__(self, clip_backend=None, bioclip_backend=None) -> None:
        # Env-driven backend selection if not injected
        if clip_backend is None or bioclip_backend is None:
            def build_backend(prefix: str, is_bio: bool = False):
                backend_name = (os.getenv(f"{prefix}_BACKEND", "torch") or "torch").lower()
                device_pref = os.getenv(f"{prefix}_DEVICE")
                max_bs_env = os.getenv(f"{prefix}_MAX_BATCH_SIZE")
                max_bs = int(max_bs_env) if max_bs_env and max_bs_env.isdigit() else None

                if backend_name == "onnxrt":
                    onnx_image = os.getenv(f"{prefix}_ONNX_IMAGE")
                    onnx_text = os.getenv(f"{prefix}_ONNX_TEXT")
                    providers = os.getenv(f"{prefix}_ORT_PROVIDERS")
                    providers_list = [p.strip() for p in providers.split(",")] if providers else None
                    return ONNXRTBackend(
                        model_name=os.getenv(f"{prefix}_MODEL_NAME"),
                        pretrained=os.getenv(f"{prefix}_PRETRAINED"),
                        onnx_image_path=onnx_image,
                        onnx_text_path=onnx_text,
                        providers=providers_list,
                        device_preference=device_pref,
                        max_batch_size=max_bs,
                    )
                if backend_name == "rknn":
                    rknn_path = os.getenv(f"{prefix}_RKNN_MODEL")
                    target = os.getenv(f"{prefix}_RKNN_TARGET", "rk3588")
                    return RKNNBackend(
                        model_name=os.getenv(f"{prefix}_MODEL_NAME"),
                        pretrained=os.getenv(f"{prefix}_PRETRAINED"),
                        rknn_model_path=rknn_path,
                        target=target,
                        device_preference=device_pref,
                        max_batch_size=max_bs,
                    )
                # default: torch backend
                model_name = os.getenv(f"{prefix}_MODEL_NAME")
                pretrained = os.getenv(f"{prefix}_PRETRAINED")
                if is_bio and not model_name:
                    model_name = "hf-hub:imageomics/bioclip-2"
                if not is_bio and not model_name:
                    model_name = "ViT-B-32"
                if not is_bio and not pretrained:
                    pretrained = "laion2b_s34b_b79k"
                return TorchBackend(
                    model_name=model_name,
                    pretrained=pretrained,
                    device_preference=device_pref,
                    max_batch_size=max_bs,
                )

            if clip_backend is None:
                clip_backend = build_backend("CLIP", is_bio=False)
            if bioclip_backend is None:
                # Allow BIOCLIP_BACKEND to fall back to CLIP_BACKEND if unset
                clip_backend_env = os.getenv("CLIP_BACKEND")
                if os.getenv("BIOCLIP_BACKEND") is None and clip_backend_env is not None:
                    os.environ["BIOCLIP_BACKEND"] = clip_backend_env
                bioclip_backend = build_backend("BIOCLIP", is_bio=True)

        # Initialize model managers with the selected backends
        self.clip_model = CLIPModelManager(backend=clip_backend)
        self.bioclip_model = BioCLIPModelManager(backend=bioclip_backend)
        self.device = self.clip_model.device  # Both models should use the same device

    def initialize(self) -> None:
        """Initializes both underlying model managers."""
        logger.info("Initializing CLIPModelManager...")
        self.clip_model.initialize()
        logger.info("Initializing BioCLIPModelManager...")
        self.bioclip_model.initialize()
        logger.info("✅ All models initialized successfully.")

    def Infer(self, request_iterator: Iterable[pb.InferRequest], context: grpc.ServicerContext):
        """
        Handles bidirectional streaming inference with server-side batching.
        """
        if not self.clip_model.is_initialized or not self.bioclip_model.is_initialized:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Models are not initialized.")
            return

        batch: List[pb.InferRequest] = []
        for req in request_iterator:
            batch.append(req)
            if len(batch) >= BATCH_SIZE:
                for response in self._process_batch(batch):
                    yield response
                batch.clear()

        # Process any remaining requests in the final, possibly smaller, batch
        if batch:
            for response in self._process_batch(batch):
                yield response

    def _process_batch(self, batch: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        """Processes a batch of requests, groups them by task, and yields responses."""
        t0 = time.time()
        logger.info(f"Processing a batch of {len(batch)} requests...")

        # Group requests by their task name
        tasks = defaultdict(list)
        for req in batch:
            tasks[req.task].append(req)

        # Process each task group
        for task_name, requests in tasks.items():
            handler = getattr(self, f"_handle_{task_name}", None)
            if handler:
                try:
                    for response in handler(requests):
                        yield response
                except Exception as e:
                    logger.exception(f"Error processing task '{task_name}': {e}")
                    # Yield an error response for all failed requests in this group
                    for req in requests:
                        yield pb.InferResponse(
                            correlation_id=req.correlation_id, is_final=True,
                            error=pb.Error(code=pb.ERROR_CODE_INTERNAL, message=str(e))
                        )
            else:
                logger.warning(f"Unknown task received: {task_name}")
                for req in requests:
                    yield pb.InferResponse(
                        correlation_id=req.correlation_id, is_final=True,
                        error=pb.Error(code=pb.ERROR_CODE_INVALID_ARGUMENT, message=f"Unknown task: {task_name}")
                    )

        processing_time = (time.time() - t0) * 1000
        logger.info(f"Batch processing finished in {processing_time:.2f} ms.")

    # --- Task Handlers (process lists of requests) ---

    def _handle_clip_classify(self, requests: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        assert self.clip_model.text_embeddings is not None
        text_features = self.clip_model.text_embeddings.to(self.device)
        for req in requests:
            with torch.no_grad():
                img_vec = torch.tensor(self.clip_model.encode_image(req.payload), device=self.device).unsqueeze(0)
                sims = (100.0 * img_vec @ text_features.T).softmax(dim=-1).squeeze(0)
                top_k = int(req.meta.get("topk", "5"))
                probs, idxs = sims.topk(top_k)
                scores = [(self.clip_model.labels[idx], float(prob)) for prob, idx in zip(probs, idxs)]
            yield self._build_label_response(req.correlation_id, scores, self.clip_model.info())

    def _handle_bioclip_classify(self, requests: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        assert self.bioclip_model.text_embeddings is not None
        text_features = self.bioclip_model.text_embeddings.to(self.device)
        for req in requests:
            with torch.no_grad():
                img_vec = torch.tensor(self.bioclip_model.encode_image(req.payload), device=self.device).unsqueeze(0)
                sims = (img_vec @ text_features.T).softmax(dim=-1).squeeze(0)
                top_k = int(req.meta.get("topk", "3"))
                probs, idxs = sims.topk(top_k)
                scores = [(self.bioclip_model.extract_scientific_name(self.bioclip_model.labels[idx]), float(prob)) for prob, idx in zip(probs, idxs)]
            yield self._build_label_response(req.correlation_id, scores, self.bioclip_model.info())

    def _handle_smart_classify(self, requests: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        assert self.clip_model.scene_prompt_embeddings is not None
        for req in requests:
            # 1) Scene classification using CLIP scene prompts
            with torch.no_grad():
                img_vec_clip = torch.tensor(self.clip_model.encode_image(req.payload), device='cpu').unsqueeze(0)
                scene_sims = (img_vec_clip @ self.clip_model.scene_prompt_embeddings.T).softmax(-1).squeeze(0)
                best_scene_idx = int(scene_sims.argmax().item())
            scene_label = self.clip_model.scene_prompts[best_scene_idx]
            is_animal_like = ("animal" in scene_label) or ("bird" in scene_label) or ("insect" in scene_label)
            if not is_animal_like:
                scene_score = float(scene_sims[best_scene_idx].item())
                yield self._build_label_response(req.correlation_id, [(scene_label, scene_score)],
                                                 self.clip_model.info(), meta={"source": "scene_classification"})
                continue
            # 2) Animal-like: classify with BioCLIP
            assert self.bioclip_model.text_embeddings is not None
            text_features = self.bioclip_model.text_embeddings.to(self.device)
            with torch.no_grad():
                img_vec_bio = torch.tensor(self.bioclip_model.encode_image(req.payload), device=self.device).unsqueeze(0)
                sims = (img_vec_bio @ text_features.T).softmax(dim=-1).squeeze(0)
                top_k = int(req.meta.get("topk", "3"))
                probs, idxs = sims.topk(top_k)
                scores = [(self.bioclip_model.extract_scientific_name(self.bioclip_model.labels[idx]), float(prob)) for prob, idx in zip(probs, idxs)]
            yield self._build_label_response(req.correlation_id, scores, self.bioclip_model.info(),
                                             meta={"source": "bioclip_classification"})

    def _handle_clip_embed(self, requests: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        for req in requests:
            vec = self.clip_model.encode_text(req.payload.decode("utf-8")).tolist()
            yield self._build_embed_response(req.correlation_id, vec, self.clip_model.info())

    def _handle_bioclip_embed(self, requests: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        for req in requests:
            vec = self.bioclip_model.encode_text(req.payload.decode("utf-8")).tolist()
            yield self._build_embed_response(req.correlation_id, vec, self.bioclip_model.info())

    def _handle_clip_image_embed(self, requests: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        # Use backend batch embedding (falls back to sequential in backend if not supported)
        image_bytes = [req.payload for req in requests]
        vecs = self.clip_model.backend.image_batch_to_vectors(image_bytes)
        for i, req in enumerate(requests):
            vec = vecs[i].tolist()
            yield self._build_embed_response(req.correlation_id, vec, self.clip_model.info())

    def _handle_bioclip_image_embed(self, requests: List[pb.InferRequest]) -> Iterable[pb.InferResponse]:
        # Use backend batch embedding (falls back to sequential in backend if not supported)
        image_bytes = [req.payload for req in requests]
        vecs = self.bioclip_model.backend.image_batch_to_vectors(image_bytes)
        for i, req in enumerate(requests):
            vec = vecs[i].tolist()
            yield self._build_embed_response(req.correlation_id, vec, self.bioclip_model.info())

    # --- Response Builders ---

    def _build_label_response(self, cid: str, scores: List[Tuple[str, float]], model_info: dict,
                              meta: dict | None = None) -> pb.InferResponse:
        model_id = f"{model_info.get('model_name', model_info.get('model_version'))}:{model_info.get('pretrained', '')}"
        obj = {"labels": [{"label": label, "score": score} for label, score in scores], "model_id": model_id.strip(":")}
        response_meta = {"labels_count": str(len(scores))}
        if meta:
            response_meta.update(meta)
        return pb.InferResponse(
            correlation_id=cid, is_final=True,
            result=json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            result_mime="application/json;schema=labels_v1",
            meta=response_meta
        )

    def _build_embed_response(self, cid: str, vec: list, model_info: dict) -> pb.InferResponse:
        model_id = f"{model_info.get('model_name', model_info.get('model_version'))}:{model_info.get('pretrained', '')}"
        obj = {"vector": vec, "dim": len(vec), "model_id": model_id.strip(":")}
        return pb.InferResponse(
            correlation_id=cid, is_final=True,
            result=json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            result_mime="application/json;schema=embedding_v1",
            meta={"dim": str(len(vec))}
        )

    # --- Capabilities and Health ---

    def GetCapabilities(self, request, context) -> pb.Capability:
        return self._build_capability()

    def StreamCapabilities(self, request, context) -> Iterable[pb.Capability]:
        yield self._build_capability()

    def Health(self, request, context):
        return empty_pb2.Empty()

    def _build_capability(self) -> pb.Capability:
        tasks = [
            pb.IOTask(name="clip_classify", input_mimes=["image/jpeg", "image/png", "image/webp"],
                      output_mimes=["application/json;schema=labels_v1"]),
            pb.IOTask(name="clip_embed", input_mimes=["text/plain"],
                      output_mimes=["application/json;schema=embedding_v1"]),
            pb.IOTask(name="bioclip_classify", input_mimes=["image/jpeg", "image/png", "image/webp"],
                      output_mimes=["application/json;schema=labels_v1"]),
            pb.IOTask(name="bioclip_embed", input_mimes=["text/plain"],
                      output_mimes=["application/json;schema=embedding_v1"]),
            pb.IOTask(name="smart_classify", input_mimes=["image/jpeg", "image/png", "image/webp"],
                      output_mimes=["application/json;schema=labels_v1"]),
            pb.IOTask(name="clip_image_embed", input_mimes=["image/jpeg", "image/png", "image/webp"],
                      output_mimes=["application/json;schema=embedding_v1"]),
            pb.IOTask(name="bioclip_image_embed", input_mimes=["image/jpeg", "image/png", "image/webp"],
                      output_mimes=["application/json;schema=embedding_v1"]),
        ]
        # Report effective runtime; if mixed backends are used, declare "mixed"
        clip_rt = self.clip_model.backend.get_info().runtime
        bio_rt = self.bioclip_model.backend.get_info().runtime
        effective_runtime = clip_rt if clip_rt == bio_rt else "mixed"
        return pb.Capability(
            service_name=self.SERVICE_NAME,
            model_ids=[self.clip_model.model_id, self.bioclip_model.model_id],
            runtime=effective_runtime,
            max_concurrency=16,
            tasks=tasks,
            extra={
                "batch_size": str(BATCH_SIZE),
                "runtime.clip": clip_rt,
                "runtime.bioclip": bio_rt,
            }
        )
