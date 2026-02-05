# Lumen Resources

Lightweight tooling for shipping Lumen ML services. This package centralizes how models are described, validated, downloaded, and cached so every service (CLIP, face, etc.) follows the same playbook—whether weights live on Hugging Face, ModelScope, or a private registry.

## Why use it?

- **Single source of truth** – YAML configs describing deployments, devices, runtimes, and model aliases.
- **Schema-backed validation** – JSON Schema plus Pydantic to catch errors before runtime.
- **Cross-platform downloads** – Intelligent routing between Hugging Face and ModelScope with caching/resume support.
- **CLI + Python API** – Automate in CI or embed in service bootstraps.
- **Result schemas** – Typed response validators (`EmbeddingV1`, `FaceV1`, `LabelsV1`) for downstream services.

## Installation

```bash
# project install
pip install "lumen-resources @ git+https://github.com/EdwinZhanCN/Lumen.git@main#subdirectory=lumen-resources"

# dev install
git clone https://github.com/EdwinZhanCN/Lumen.git
cd Lumen/lumen-resources
pip install -e ".[dev,config]"
```

Optional extras depending on your targets:

```bash
pip install huggingface_hub
pip install modelscope
pip install torch torchvision
pip install onnxruntime
```

## Usage

### CLI

```bash
# download everything defined in config.yaml
lumen-resources download config.yaml

# strict config validation
lumen-resources validate config.yaml

# validate a model_info.json
lumen-resources validate-model-info path/to/model_info.json

# inspect cache contents (defaults to ~/.lumen/)
lumen-resources list ~/.lumen/
```

### Python API

```python
from lumen_resources import (
    load_and_validate_config,
    Downloader,
    load_and_validate_model_info,
    EmbeddingV1,
)

config = load_and_validate_config("config.yaml")
downloader = Downloader(config, verbose=True)
results = downloader.download_all(force=False)

model_info = load_and_validate_model_info("model_info.json")
```

## Configuration essentials

```yaml
metadata:
  region: "other"      # or "cn" to prefer ModelScope
  cache_dir: "~/.lumen/models"

deployment:
  mode: "single"       # or "hub"
  service: "clip"

services:
  clip:
    enabled: true
    package: "lumen_clip"
    backend_settings:
      device: "cuda"
      batch_size: 16
      onnx_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    models:
      default:
        model: "ViT-B-32"
        runtime: "torch"
      fp16:
        model: "ViT-B-32"
        runtime: "onnx"
```

- `metadata.region` decides whether downloads prefer ModelScope or Hugging Face.
- `backend_settings` lets you declare execution providers, batch sizes, devices, etc.
- Each entry in `models` becomes a cache namespace (`clip/default`, `clip/fp16`, …).

## Reference

- Source: `src/lumen_resources/`
  - `lumen_config.py` – Typed config models
  - `downloader.py` – Platform abstraction + caching
  - `cli.py` – Command entrypoint
  - `result_schemas/` – Response validators
- Docs: https://doc.lumilio.org
- Issues & support: open a ticket in the main Lumen monorepo.