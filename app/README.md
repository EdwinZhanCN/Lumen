# Lumen Configuration Manager

A Tauri-based desktop application for creating and managing Lumen ML service configurations.

## 🎯 Overview

This application provides a GUI for creating `lumen.yaml` configuration files for Lumen ML services (Face, CLIP, OCR, VLM). Each service reads only its own section from the config file, so a single `lumen.yaml` can contain multiple services.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- Rust 1.70+
- System dependencies for Tauri (see [Tauri docs](https://tauri.app/v2/guides/prerequisites))

### Development

```bash
# Install dependencies
npm install

# Run in development mode (hot reload)
npm run tauri dev

# Build for production
npm run tauri build
```

## 📋 Features

### Configuration Builder
- **Metadata**: Version, region (HuggingFace/ModelScope), cache directory with folder picker
- **Deployment Mode**: Single (one service per process) or Hub (all services in one process)
- **Server**: Host, port, Lumen AI Mesh (automatic service discovery)
- **Services**: Add/remove services (Face, CLIP, OCR)
  - Simplified hardware selection (NVIDIA GPU, Apple Silicon, CPU, Intel GPU, Rockchip NPU)
  - Automatic ONNX provider configuration with CPU fallback
  - Model configuration per service
  - Enable/disable services
  - Advanced settings toggle - manual control of device, batch size, and ONNX providers
  - **Single mode**: Only one service allowed per config
  - **Hub mode**: Multiple services allowed (coming soon)

### Real-Time Preview
- Live YAML preview as you edit
- Copy to clipboard
- Download as file
- Save to disk (`~/.lumen/lumen.yaml`)

### User-Friendly Design
- Native file explorer integration for cache directory selection (macOS Finder / Windows Explorer)
- Simplified hardware configuration - no need to understand ONNX, CUDA, or CoreML
- Technical details hidden in tooltips and collapsible sections
- Progressive disclosure: Advanced settings toggle for manual control
- Mutually exclusive modes: Use simplified device types OR advanced manual configuration

## 🏗️ Architecture

```
app/
├── src/                          # React frontend
│   ├── components/
│   │   ├── ConfigBuilder/        # Configuration form components
│   │   └── YamlPreview/          # YAML preview component
│   ├── utils/
│   │   ├── yaml.ts               # YAML serialization & validation
│   │   ├── tauri.ts              # Tauri API wrappers
│   │   └── defaultConfig.ts      # Templates & constants
│   └── types/
│       └── lumen-config.d.ts     # Generated from JSON Schema
│
└── src-tauri/                    # Rust backend
    └── src/
        ├── commands/             # Tauri commands
        └── config/               # File operations

```

## 📝 Configuration File

The app generates a single `lumen.yaml` file at `~/.lumen/lumen.yaml`. 

### Deployment Modes

**Single Mode (Current)**:
- Each service runs in its own Python process
- Each service reads only its own section from the config
- Each service has its own gRPC port
- Only one service per config file (UI enforced)
- Run with: `python -m lumen_face --config ~/.lumen/lumen.yaml`

**Hub Mode (Coming Soon)**:
- All services run in a single Python process via `hub.py`
- One gRPC port for all services
- Multiple services allowed in one config
- Lower memory footprint

Example structure:

```yaml
metadata:
  version: "1.0.0"
  region: other
  cache_dir: "~/.lumen"

deployment:
  mode: single  # or 'hub' (coming soon)

server:
  port: 50051  # Each service uses its own port in single mode
  host: "0.0.0.0"

services:
  face:
    enabled: true
    package: "lumen_face"
    import:
      registry_class: "lumen_face.service.FaceServiceRegistry"
      add_to_server: "lumen_face.service.add_to_server"
    backend_settings:
      device: cuda  # Auto-configured based on hardware selection
      batch_size: 1
      onnx_providers:
        - CUDAExecutionProvider
        - CPUExecutionProvider  # CPU fallback for all device types
    models:
      default:
        model: "buffalo_l"
        runtime: onnx

  # Note: In single mode, only configure ONE service.
  # For multiple services, use separate config files or wait for hub mode.
  # clip:
  #   enabled: true
  #   package: "lumen_clip"
  #   ...
```

## 🔧 Tech Stack

- **Frontend**: React 19 + TypeScript + Tailwind CSS + daisyUI
- **Backend**: Rust + Tauri 2
- **Build**: Vite

## 🎨 UI/UX Design

- **Progressive Disclosure**: Technical details hidden by default, accessible when needed
- **Tooltips**: Hover info icons for contextual help without overwhelming the interface
- **Smart Defaults**: Automatically configures ONNX providers based on hardware selection with CPU fallback
- **Mutually Exclusive Modes**: Device Type auto-configuration OR Advanced Settings manual control
- **Terminology**: 
  - "Lumen AI Mesh" instead of "mDNS"
  - "Hardware Configuration" instead of "Backend Settings"
  - Device types (NVIDIA GPU, Apple Silicon, CPU, Intel GPU) instead of technical terms (CUDA, MPS, OpenVINO)
- Tailwind CSS 4 for styling
- daisyUI for component library
- Type-safe with auto-generated TypeScript types from JSON Schema

## 📦 Type Generation

Types are automatically generated from the JSON Schema:

```bash
npx json-schema-to-typescript \
  --input ../lumen-resources/src/lumen_resources/schemas/config-schema.yaml \
  --output src/types/lumen-config.d.ts
```

## 📄 License

[Same as parent Lumen project]

---

**Version**: 0.1.0