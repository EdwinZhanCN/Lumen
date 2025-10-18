最佳解决方案：分离文件，各司其职

最干净、最能同时满足两个目标的解决方案，就是将它们分离到两个独立的文件中。这完全符合“职责单一”的原则。

1.  **`ImageNet_1k_embeddings.npy`**: 一个只包含 `float32` 嵌入向量的巨大文件。它是内存映射的主角。
2.  **`ImageNet_1k_labels.json`**: 一个轻量的、包含字符串标签列表的JSON文件。它很小，一次性加载到内存中没有任何问题。

为了让 `ResourceLoader` 知道这个约定，我们需要对 `model_info.json` 的 `datasets` 结构做一个小小的扩展。

---

### 实施步骤

#### 1. 更新 `model_info.json` 结构

我们将 `datasets` 中每个条目的值从一个**字符串**（文件名）升级为一个**对象**，该对象明确指明了标签和嵌入向量文件的路径。

**旧结构**:
```json
"datasets": {
  "ImageNet_1k": "ImageNet_1k.npz"
}
```

**新结构**:
```json
"datasets": {
  "ImageNet_1k": {
    "labels": "ImageNet_1k_labels.json",
    "embeddings": "ImageNet_1k_embeddings.npy"
  }
}
```
这个新结构更加清晰和可扩展。

#### 2. 修改 `_load_dataset` 以支持新结构和 `mmap_mode`

现在，我们重写 `resources/loader.py` 中的 `_load_dataset` 函数，让它能够理解这个新结构，并在加载 `.npy` 文件时启用 `mmap_mode`。

**请用下面的代码替换 `_load_dataset` 函数**：
```python
    @staticmethod
    def _load_dataset(
        model_root_path: Path, model_info: ModelConfigurationSchema, dataset: str | None
    ) -> tuple[NDArray[np.object_] | None, NDArray[np.float32] | None]:
        if not model_info.datasets:
            logger.info("No datasets configured in model_info.json")
            return None, None

        dataset_name = dataset
        if not dataset_name:
            if model_info.model_type == "clip":
                dataset_name = "ImageNet_1k"
            elif model_info.model_type == "bioclip":
                dataset_name = "TreeOfLife-10M"

        if not dataset_name or dataset_name not in model_info.datasets:
            logger.warning(f"Dataset '{dataset_name}' not in model_info.json. Disabling classification.")
            return None, None

        dataset_info = model_info.datasets[dataset_name]

        try:
            # --- NEW LOGIC FOR MEMORY MAPPING ---

            # Case 1: New two-file structure (dict with 'labels' and 'embeddings')
            if isinstance(dataset_info, dict):
                labels_path = model_root_path / dataset_info["labels"]
                embeddings_path = model_root_path / dataset_info["embeddings"]

                if not labels_path.exists() or not embeddings_path.exists():
                    raise FileNotFoundError(f"Missing dataset files: {labels_path} or {embeddings_path}")

                # Load labels from JSON (or other simple formats)
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels_list = json.load(f)
                labels = np.array(labels_list, dtype=object)

                # Load embeddings using memory mapping
                logger.info(f"Loading embeddings for '{dataset_name}' with memory mapping...")
                embeddings = np.load(embeddings_path, mmap_mode='r')

                logger.info(f"Loaded dataset '{dataset_name}': {len(labels)} classes")
                return labels, embeddings

            # Case 2: Legacy single-file structure (string filename)
            elif isinstance(dataset_info, str):
                logger.warning("Using legacy single-file dataset format. Memory mapping is not guaranteed.")
                dataset_path = model_root_path / dataset_info
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

                data = np.load(dataset_path, allow_pickle=True)
                # ... (old logic for npz/npy remains here for backward compatibility)
                if isinstance(data, np.lib.npyio.NpzFile):
                    labels = data["labels"]
                    embeddings = data.get("embeddings")
                    data.close()
                else: # .npy with a dict inside
                    data_dict = data.item()
                    labels = np.array(data_dict["labels"], dtype=object)
                    embeddings = data_dict.get("embeddings")

                logger.info(f"Loaded dataset '{dataset_name}': {len(labels)} classes")
                return labels, embeddings

            else:
                raise TypeError(f"Unsupported format for dataset info: {type(dataset_info)}")

        except Exception as e:
            logger.warning(f"Failed to load dataset '{dataset_name}': {e}. Disabling classification.")
            return None, None
```

### 总结

这个方案是两全其美的最佳实践：

1.  **实现了您的性能目标**：通过分离文件，我们为巨大的 `embeddings` 数组启用了 `mmap_mode='r'`，显著优化了内存和启动性能。
2.  **代码和配置更清晰**：`model_info.json` 的新结构明确地描述了需要哪些文件，代码也相应地变得更加健壮，能够处理多种数据格式。
3.  **保持了向后兼容**：代码仍然可以处理旧的单文件 `.npz` 格式，确保了系统的平滑过渡。

您提出的关于 `mmap_mode` 的问题非常有价值，它将我们的资源加载器提升到了一个新的健壮性和性能水平。
