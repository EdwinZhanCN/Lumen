# Lumen-CLIP (PML: Python Machine Learning)

pml stands for “Python Machine Learning.” This package provides a unified, high-performance gRPC inference service for Lumilio Photos. It exposes both general-purpose CLIP and BioCLIP-2 capabilities for image classification and text/image embeddings, with batched request processing and optional mDNS advertisement for discovery on a local network.

- Status:
  - image_classification: implemented (OpenCLIP + ImageNet labels, scene prompts, text/image embeddings)
  - biological_atlas: implemented (BioCLIP-2 + TreeOfLife-10M labels, text/image embeddings)

## Project structure

```
lumen-clip/
├─ Dockerfiles/
│  ├─ Dockerfile                # CPU (and macOS with MPS) target
│  ├─ cuda/
│  │  └─ Dockerfile             # NVIDIA CUDA (12.6) target
│  └─ rocm/
│     └─ Dockerfile             # AMD ROCm (6.3) target
├─ data/                        # Runtime caches (labels & precomputed text vectors)
│  ├─ clip/
│  └─ bioclip/
├─ src/
│  ├─ server.py                 # Unified gRPC server (UnifiedMLService + mDNS)
│  ├─ proto/
│  │  ├─ ml_service.proto       # Inference protocol
│  │  ├─ ml_service_pb2.py      # Generated code
│  │  └─ ml_service_pb2_grpc.py # Generated code
│  ├─ service_registry/
│  │  ├─ __init__.py
│  │  └─ service.py             # UnifiedMLService (task router + batching)
│  ├─ image_classification/
│  │  ├─ clip_model.py          # CLIPModelManager (OpenCLIP, ImageNet labels, scene prompts)
│  │  └─ clip_service.py        # Standalone CLIP service (reference/legacy)
│  └─ biological_atlas/
│     ├─ bioclip_model.py       # BioCLIPModelManager (OpenCLIP, TreeOfLife-10M labels)
│     └─ bioclip_service.py     # Standalone BioCLIP service (reference/legacy)
├─ pyproject.toml               # Python packaging & deps
├─ uv.lock                      # uv lock file
└─ README.md
```

Notes:
- The unified server in `src/server.py` starts `UnifiedMLService` (see `src/service_registry/service.py`).
- Caches:
  - CLIP: downloads ImageNet-1k labels and precomputes text embeddings to `data/clip`.
  - BioCLIP: downloads TreeOfLife-10M labels and precomputes text embeddings to `data/bioclip`.
- The standalone `clip_service.py` and `bioclip_service.py` implement the same proto one-service-per-model; they are kept as references but are not started by the unified server.


## gRPC services and tasks

Protocol definition: `src/proto/ml_service.proto`

Service: `service Inference`
- rpc Infer(stream InferRequest) returns (stream InferResponse)
- rpc GetCapabilities(google.protobuf.Empty) returns (Capability)
- rpc StreamCapabilities(google.protobuf.Empty) returns (stream Capability)
- rpc Health(google.protobuf.Empty) returns (google.protobuf.Empty)

UnifiedMLService supported tasks (name → input → output):
- clip_classify
  - input: image/jpeg | image/png | image/webp
  - output: application/json;schema=labels_v1
  - meta: topk (default 5)
- bioclip_classify
  - input: image/jpeg | image/png | image/webp
  - output: application/json;schema=labels_v1
  - meta: topk (default 3)
- smart_classify
  - input: image/jpeg | image/png | image/webp
  - output: application/json;schema=labels_v1
  - behavior: CLIP scene prompts first; if “animal-like”, delegates to BioCLIP classification, otherwise returns the predicted scene
- clip_embed
  - input: text/plain
  - output: application/json;schema=embedding_v1
- bioclip_embed
  - input: text/plain
  - output: application/json;schema=embedding_v1
- clip_image_embed
  - input: image/jpeg | image/png | image/webp
  - output: application/json;schema=embedding_v1
- bioclip_image_embed
  - input: image/jpeg | image/png | image/webp
  - output: application/json;schema=embedding_v1

Batching:
- Unified service processes requests in batches (default batch size 8) for higher throughput. Image-embed tasks use a true batched forward pass; other tasks batch at the routing level.

Capabilities and health:
- GetCapabilities/StreamCapabilities describe the above tasks and limits.
- Health returns Empty when the service is healthy.

mDNS advertisement:
- If zeroconf is available, the server publishes `_homenative-node._tcp.local.` with service name `CLIP-Image-Proccesor._homenative-node._tcp.local.` and TXT props:
  - uuid: CLIP_MDNS_UUID or randomly generated
  - status: CLIP_MDNS_STATUS (default “ready”)
  - version: CLIP_MDNS_VERSION (default “1.0.0”)
- You can override the advertised IP via `ADVERTISE_IP` env var. If it resolves to a loopback address (127.x), remote discovery may fail.


## Installation

This project uses uv for dependency management.

- Python version for local development: see pyproject “requires-python” (>=3.12,<3.13). If you prefer newer Python, use the Docker images below which include a pinned runtime.

- Apple Silicon (macOS, MPS):
````bash
uv pip install '.[osx]'
````

- NVIDIA GPU (CUDA 12.6):
````bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 '.[gpu]'
````

- CPU-only:
````bash
uv pip install '.[cpu]'
````


## Running locally

Initialize environment and start the unified server:
````bash
# From the project root
export PYTHONPATH=$(pwd)/src

# Optional: control mDNS advertising
export ADVERTISE_IP=192.168.1.23
export CLIP_MDNS_UUID="$(uuidgen)"
export CLIP_MDNS_STATUS=ready
export CLIP_MDNS_VERSION=1.0.0

# Start the server
python -m src.server --port 50051
````

Quick checks with grpcurl:
````bash
# List services
grpcurl -plaintext localhost:50051 list

# Health
grpcurl -plaintext localhost:50051 home_native.v1.Inference/Health

# Capabilities (single response)
grpcurl -plaintext localhost:50051 home_native.v1.Inference/GetCapabilities
````

Example request/response shapes
- InferRequest
  - fields: correlation_id, task, payload (bytes), payload_mime, meta (map), seq/total/offset (optional for chunking)
- InferResponse
  - fields: correlation_id, is_final, result (bytes), result_mime, meta (map), error (optional), seq/total/offset

Example: clip_classify (labels_v1)
````json
{
  "labels": [
    {"label": "golden_retriever", "score": 0.73},
    {"label": "Labrador_retriever", "score": 0.14}
  ],
  "model_id": "ViT-B-32:laion2b_s34b_b79k"
}
````

Example: clip_embed / bioclip_embed (embedding_v1)
````json
{
  "vector": [0.01, -0.02, ...],
  "dim": 512,
  "model_id": "ViT-B-32:laion2b_s34b_b79k"
}
````

Note: The exact embedding dimension varies by model; it is reported in the payload and duplicated in response meta as dim.


## Docker

Multiple targets are supplied to match different runtimes.

- CPU (and macOS MPS) image:
````bash
docker build -f Dockerfiles/Dockerfile -t lumen-clip:cpu .
docker run --rm -p 50051:50051 \
  -e ADVERTISE_IP \
  -e CLIP_MDNS_UUID \
  -e CLIP_MDNS_STATUS=ready \
  -e CLIP_MDNS_VERSION=1.0.0 \
  lumen-clip:cpu python -m src.server --port 50051
````

- NVIDIA CUDA 12.6 image:
````bash
docker build -f Dockerfiles/cuda/Dockerfile -t lumen-clip:cuda .
docker run --rm --gpus all -p 50051:50051 \
  -e ADVERTISE_IP \
  -e CLIP_MDNS_UUID \
  -e CLIP_MDNS_STATUS=ready \
  -e CLIP_MDNS_VERSION=1.0.0 \
  lumen-clip:cuda python -m src.server --port 50051
````

- AMD ROCm 6.3 image:
````bash
docker build -f Dockerfiles/rocm/Dockerfile -t lumen-clip:rocm .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -p 50051:50051 \
  -e ADVERTISE_IP \
  -e CLIP_MDNS_UUID \
  -e CLIP_MDNS_STATUS=ready \
  -e CLIP_MDNS_VERSION=1.0.0 \
  lumen-clip:rocm python3 -m src.server --port 50051
````

Notes:
- The Docker build stages install dependencies with uv. The runtime entrypoint above uses `python -m src.server` to match the source layout (`src/server.py`).
- The images copy the project into `/app` and set `PYTHONPATH=/app`. Adjust bind mounts and volumes as needed to persist `data/*` caches between runs.


## Development workflow

- Model managers
  - `CLIPModelManager` uses OpenCLIP (e.g., `ViT-B-32:laion2b_s34b_b79k`) and caches:
    - ImageNet labels: `data/clip/imagenet_class_names.json` (downloaded from Hugging Face)
    - Text embeddings: `data/clip/<model>_imagenet_vectors.npz`
  - `BioCLIPModelManager` uses `imageomics/bioclip-2` and caches:
    - TreeOfLife-10M labels: `data/bioclip/txt_emb_species.json`
    - Text embeddings: `data/bioclip/text_vectors.npz`
- Unified service
  - Routes tasks and applies server-side batching where possible.
  - High-performance image preprocessing uses `torchvision.io.decode_image` with a fallback to Pillow for less common formats (e.g., WebP).
  - Scene routing in `smart_classify`: CLIP scene prompts → if animal-like, delegate to BioCLIP.
- Regenerating gRPC stubs (only if you modify `ml_service.proto`):
````bash
python -m grpc_tools.protoc \
  -I src/proto \
  --python_out=src/proto \
  --grpc_python_out=src/proto \
  src/proto/ml_service.proto
````
- Python version
  - Local dev targets Python 3.12 (see pyproject). Use the Docker targets if you prefer a pinned container runtime.
- Logging
  - The server logs to stdout; additional logs can be directed as needed. You may see `pml_service.log` if you configure file-based logging.
- Performance tuning
  - Batch size is currently fixed at 8 in the unified service. Increase cautiously based on device memory and throughput requirements.


## Data sources and caching

- ImageNet labels are downloaded once from: https://huggingface.co/datasets/huggingface/label-files (file: `imagenet-1k-id2label.json`) and stored at `data/clip/imagenet_class_names.json`.
- TreeOfLife-10M species labels are downloaded once from: https://huggingface.co/datasets/imageomics/TreeOfLife-10M (file: `embeddings/txt_emb_species.json`) and stored at `data/bioclip/txt_emb_species.json`.
- First run computes and caches text embeddings for label sets. Subsequent runs load from cache for faster startup.

If you need to rebuild label vectors (e.g., after changing models), delete the corresponding `*.npz` files in `data/clip` or `data/bioclip`.


## Citations and attributions

If you use this project in academic work or presentations, please cite the upstream models and datasets:

- OpenAI CLIP
  - Paper: Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)
  - BibTeX:
````bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamila and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021}
}
````
- open-clip-torch
  - Repository: https://github.com/mlfoundations/open_clip
  - Please follow repository guidance for appropriate citations and dataset acknowledgements (e.g., LAION).
- BioCLIP 2
  - The project uses BioCLIP-2 for biological atlas tasks.
  - BibTeX (as provided by the authors):
````bibtex
@article{gu2025bioclip,
  title = {{B}io{CLIP} 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2025},
  eprint={2505.23883},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.23883},
}
````
- TreeOfLife-10M (label source used by BioCLIP-2 manager)
  - Dataset (Hugging Face): https://huggingface.co/datasets/imageomics/TreeOfLife-10M

Additionally:
- This project uses PyTorch, TorchVision, Pillow, gRPC, protobuf, zeroconf, and the Hugging Face Hub. Please cite them where appropriate per their respective guidelines.


## Troubleshooting

- Discovery issues on LAN
  - If mDNS advertises a loopback address (127.x), set `ADVERTISE_IP` to a reachable LAN IP.
- GPU initialization
  - NVIDIA: ensure host drivers match the CUDA 12.6 base image, and use `--gpus all`.
  - ROCm: ensure the host supports ROCm 6.3 and pass `/dev/kfd` and `/dev/dri` devices to the container.
- Cold start
  - First run downloads labels and computes text embeddings; this can take several minutes depending on hardware. Subsequent runs use cached vectors.
- Protocol compatibility
  - Use the provided proto to generate stubs in your client language. Respect `payload_mime` and `result_mime` fields so both sides negotiate payload schemas correctly.


## License

This project is licensed under the GPL-3.0 license. See the LICENSE file for details.
