# CLIP Asset Tooling

This directory contains the CLIP-specific asset pipeline for Lumen.

The public workflow has a single CLI entry:

```bash
uv run python scripts/clip/export_clip_to_onnx.py --model-root /path/to/model-repo
```

Everything else under `scripts/common/` is an internal helper layer.

## Repository Layout

The CLIP pipeline produces this repository shape:

```text
{model_repo}/
  README.md
  model_info.json
  upstream/
    source.json
    ...
  onnx/
    vision.fp32.onnx
    vision.fp16.onnx
    text.fp32.onnx
    text.fp16.onnx
  validation/
    onnx/
      fp32.json
      fp16.json
```

There is no `lumen_metadata.json`. The readable summary lives in `README.md`,
while the runtime-facing contract lives in `model_info.json`.

## Supported Sources

### Hugging Face

Expected `upstream/source.json`:

```json
{
  "format": "huggingface",
  "repo_id": "openai/clip-vit-base-patch32",
  "revision": "main"
}
```

For Hugging Face sources, the pipeline is allowed to fetch missing CLIP assets
into `upstream/` on demand. The CLIP script decides the allow-list because the
needed files are task-specific.

### OpenCLIP

Expected `upstream/source.json`:

```json
{
  "format": "openclip",
  "repo_id": "MobileCLIP2-S4",
  "pretrained": "dfndr2b",
  "export_options": {
    "reparameterize": true
  }
}
```

Notes:

- `repo_id` maps to the first argument of
  `open_clip.create_model_and_transforms(...)`.
- `pretrained` maps to the pretrained tag passed into OpenCLIP.
- `export_options.reparameterize=true` enables
  `timm.utils.reparameterize_model()` before ONNX export.
- OpenCLIP models may not provide a standalone `tokenizer.json`. In that case,
  validation records tokenization as `runtime-bound`.

## What the Pipeline Does

`export_clip_to_onnx.py` performs the whole CLIP asset build:

1. Read `upstream/source.json`
2. Download missing Hugging Face CLIP artifacts when the source format is `huggingface`
3. Load the CLIP model through either `transformers` or `open_clip`
4. Export separate `vision` and `text` ONNX graphs
5. Convert the exported FP32 graphs into FP16 graphs
6. Generate `model_info.json`
7. Validate all ONNX artifacts
8. Generate the model repository `README.md`

## CLI

```bash
uv run python scripts/clip/export_clip_to_onnx.py \
  --model-root /path/to/model-repo \
  --model-name clip-vit-base-patch32 \
  --description "CLIP deployment assets" \
  --version 1.0.0 \
  --opset 17 \
  --image-shape 1,3,224,224 \
  --text-shape 1,77
```

Arguments:

- `--model-root`
  - Target model repository directory.
- `--model-name`
  - Optional override for `model_info.json.name`.
  - If omitted, the script derives it from `source.json.repo_id`.
- `--description`
  - Optional override for `model_info.json.description`.
- `--version`
  - Optional asset package version. Defaults to `1.0.0`.
- `--opset`
  - ONNX opset version. Defaults to `17`.
- `--image-shape`
  - Fixed image input shape. Defaults to `1,3,224,224`.
- `--text-shape`
  - Fixed text input shape. Defaults to `1,77`.

## Examples

### Hugging Face CLIP

```bash
mkdir -p /path/to/model-repo/upstream
cat >/path/to/model-repo/upstream/source.json <<'JSON'
{
  "format": "huggingface",
  "repo_id": "openai/clip-vit-base-patch32",
  "revision": "main"
}
JSON

uv run python scripts/clip/export_clip_to_onnx.py \
  --model-root /path/to/model-repo \
  --model-name clip-vit-base-patch32 \
  --description "CLIP deployment assets" \
  --image-shape 1,3,224,224 \
  --text-shape 1,77
```

### OpenCLIP with Reparameterization

```bash
mkdir -p /path/to/model-repo/upstream
cat >/path/to/model-repo/upstream/source.json <<'JSON'
{
  "format": "openclip",
  "repo_id": "MobileCLIP2-S4",
  "pretrained": "dfndr2b",
  "export_options": {
    "reparameterize": true
  }
}
JSON

uv run python scripts/clip/export_clip_to_onnx.py \
  --model-root /path/to/model-repo \
  --model-name mobileclip2-s4 \
  --description "OpenCLIP deployment assets" \
  --image-shape 1,3,224,224 \
  --text-shape 1,77
```

## Internal Helpers

The CLIP pipeline internally uses:

- `scripts/common/hf_download.py`
- `scripts/common/write_model_info.py`
- `scripts/common/validate_onnx.py`
- `scripts/common/generate_model_card.py`
- `scripts/clip/validate_clip_assets.py`

These are implementation helpers, not separate end-user workflows.
