## Lumen

### Machine Learning Module for Lumilio-Photos

### Home Native
- Compose profiles
- Zeroconf, mDNS discovery
- gRPC Unified Protobuf
- Model Factory (Not implemented yet) 🚧

### Modules
- [lumen-clip](./lumen-clip/README.md) - CLIP Service utilize open-clip for normal image embeddings and classification, bioclip2 for expert animal classification.
- [lumen-face](./lumen-face/README.md) - In progress 🚧
- [lumen-ocr](./lumen-ocr/README.md) - In progress 🚧
python -m grpc_tools.protoc \
  -I proto \
  --python_out=src \
  --grpc_python_out=src \
  proto/ml_service.proto
