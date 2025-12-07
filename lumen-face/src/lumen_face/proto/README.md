## Proto Code Gen

Before run the command, please make sure you're right in `Lumen/lumen-*/` directory.

```bash
python -m grpc_tools.protoc \
    -I src/lumen_clip \
    --python_out=src/lumen_clip \
    --grpc_python_out=src/lumen_clip \
    --pyi_out=src/lumen_clip \
    src/lumen_clip/proto/ml_service.proto
```
