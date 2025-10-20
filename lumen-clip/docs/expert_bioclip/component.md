# BioCLIP Component Diagram

This document contains a component-level overview of how the `BioCLIPModelManager` relates to its primary dependencies and consumers. Use this to quickly understand the static relationships and data flow between:

- `BioCLIPModelManager` — the business-level manager implemented in `src/lumen_clip/expert_bioclip/bioclip_model.py`
- `ModelResources` — pre-loaded labels, embeddings and model metadata
- `BaseClipBackend` — the inference backend (image/text encoding, batch support, info)
- `Client` — any caller that uses the manager (API, CLI, notebook, etc.)
- data artifacts: `labels`, `label_embeddings`

Nodes and relationships:
- `BioCLIPModelManager` "reads" from `ModelResources` to obtain `labels` and `label_embeddings`.
- `BioCLIPModelManager` "uses" the `BaseClipBackend` to initialize runtime and to encode text/images.
- `Client` invokes `BioCLIPModelManager` methods such as `initialize()`, `classify_image(...)`, `encode_image(...)`, `encode_text(...)`.
- When classification is supported but `label_embeddings` are missing, the manager requests text embeddings from the `BaseClipBackend` via batch or sequential `text_to_vector` calls.

Mermaid component diagram (paste into a Mermaid renderer or compatible viewer to visualize):

```mermaid
graph LR
  subgraph Consumer
    Client["Client\n(caller: API / CLI / Notebook)"]
  end

  subgraph ManagerModule ["BioCLIP Module"]
    Manager["BioCLIPModelManager\n- initialize()\n- classify_image()\n- encode_image()\n- encode_text()"]
  end

  subgraph Resources ["ModelResources"]
    Labels["labels (TreeOfLife)\n- list[str]"]
    Embeddings["label_embeddings\n- NDArray[np.float32]"]
    Metadata["model_name / runtime\nhas_classification_support()"]
  end

  subgraph Backend ["BaseClipBackend (runtime)"]
    BackendAPI["BaseClipBackend\n- initialize()\n- image_to_vector()\n- text_to_vector()\n- text_batch_to_vectors()\n- get_info()"]
  end

  %% Relationships
  Client -->|calls| Manager
  Manager -->|reads| Labels
  Manager -->|reads| Embeddings
  Manager -->|reads metadata| Metadata
  Manager -->|uses| BackendAPI
  Labels -->|provided by| Resources
  Embeddings -->|provided by| Resources
  Metadata -->|provided by| Resources

  %% Behavioral note as a separate node
  Note["Behavior: If classification supported and
  `label_embeddings` is None:
  Manager calls Backend.text_batch_to_vectors(prompts)
  fallback -> text_to_vector for each prompt"]
  
  Manager -.-> Note
```

Quick reference: important fields and methods to map to the diagram
- `BioCLIPModelManager.backend` — injected `BaseClipBackend`
- `BioCLIPModelManager.resources` — `ModelResources` instance
- `BioCLIPModelManager.labels` — populated from `resources.labels`
- `BioCLIPModelManager.text_embeddings` — `resources.label_embeddings` or computed via backend
- `BioCLIPModelManager.supports_classification` — `resources.has_classification_support()`
