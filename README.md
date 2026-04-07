## Lumen

### Modules
- [lumen-clip](./packages/lumen-clip/README.md)
- [lumen-face](./packages/lumen-face/README.md)
- [lumen-ocr](./packages/lumen-ocr/README.md)
- [lumen-vlm](./packages/lumen-vlm/README.md)
- [lumen-resources](./packages/lumen-resources/README.md)
- [lumen-app](./lumen-app/README.md)

### Asset Tooling
- Lumen model repositories use an `upstream/ + onnx/ + validation/` layout so raw upstream artifacts and deployment artifacts stay separate.
- `scripts/common/` contains helper functions only. Task pipelines own the actual workflow.
- `scripts/clip/` is the first task-specific reference implementation and exposes a single CLI entrypoint: [`scripts/clip/export_clip_to_onnx.py`](./scripts/clip/export_clip_to_onnx.py).
- CLIP model repositories use `model_info.json` as the runtime-facing manifest and generate a readable `README.md` instead of a separate `lumen_metadata.json`.
- See [`scripts/clip/README.md`](./scripts/clip/README.md) for the CLIP workflow.
- OpenCLIP exports can optionally apply `timm.utils.reparameterize_model()` before ONNX export when the upstream source metadata requests it.
- `lumen-resources` remains the current runtime package, but these scripts are designed so future resource handling can move under `lumen.resources`.
