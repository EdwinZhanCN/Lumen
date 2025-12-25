#!/usr/bin/env python3
"""
Compute BioCLIP text embeddings directly to NPY format.

This script creates 768-dimensional text embeddings for TreeOfLife-10M
and saves them directly as NPY file, matching BioCLIP's expectations.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to compute BioCLIP text embeddings directly to NPY."""

    # Configuration
    model_id = "hf-hub:imageomics/bioclip-2"
    text_repo_id = "imageomics/TreeOfLife-10M"
    remote_names_path = "embeddings/txt_emb_species.json"
    batch_size = 512

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "bioclip"
    data_dir.mkdir(parents=True, exist_ok=True)

    names_filename = data_dir / "txt_emb_species.json"
    embeddings_filename = data_dir / "text_vectors_768.npy"

    logger.info("Starting BioCLIP text embedding computation...")
    logger.info(f"Model: {model_id}")
    logger.info(f"Output file: {embeddings_filename}")

    # 1. Load BioCLIP model
    logger.info("Loading BioCLIP model...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    model.eval().to(device)

    # 2. Download TreeOfLife-10M labels
    logger.info("Loading TreeOfLife-10M labels...")
    if not names_filename.exists():
        logger.info("Downloading label names from HuggingFace...")
        downloaded_path = hf_hub_download(
            repo_id=text_repo_id,
            repo_type="dataset",
            filename=remote_names_path,
        )
        # Copy to our data directory
        import shutil

        shutil.copy(downloaded_path, names_filename)
        logger.info(f"Labels saved to: {names_filename}")

    # 3. Load labels
    logger.info("Loading labels...")
    with open(names_filename) as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} labels")

    # 4. Compute text embeddings in batches
    logger.info("Computing 768-dimensional text embeddings...")
    all_vecs = []

    t0 = time.time()
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i : i + batch_size]
        prompts = [f"a photo of {name}" for name in batch_labels]

        # Tokenize
        tokens = tokenizer(prompts).to(device)

        # Encode with BioCLIP
        with torch.no_grad():
            batch_features = model.encode_text(tokens)
            # Normalize to unit vectors
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

        all_vecs.append(batch_features.cpu().numpy())

        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(labels):
            logger.info(f"Processed {i + len(batch_labels)}/{len(labels)} labels...")

    # 5. Combine all embeddings and save directly as NPY
    logger.info("Combining embeddings...")
    embeddings = np.vstack(all_vecs).astype(np.float32)

    logger.info(f"Final embedding shape: {embeddings.shape}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")

    # Save directly as NPY
    logger.info(f"Saving embeddings to: {embeddings_filename}")
    np.save(embeddings_filename, embeddings)

    elapsed = time.time() - t0
    logger.info(f"✅ Completed in {elapsed:.2f} seconds")
    logger.info(f"✅ Text embeddings: {embeddings.shape}")
    logger.info(f"✅ Saved to: {embeddings_filename}")

    # Verification
    loaded_embeddings = np.load(embeddings_filename)
    logger.info("Verification:")
    logger.info(
        f"  File size: {embeddings_filename.stat().st_size / (1024 * 1024 * 1024):.2f} GB"
    )
    logger.info(f"  Sample label: {labels[0]}")
    logger.info(f"  Embedding norm: {np.linalg.norm(loaded_embeddings[0]):.6f}")
    logger.info(f"  Embedding dim: {loaded_embeddings.shape[1]}")
    logger.info(
        f"  All embeddings normalized: {np.allclose(np.linalg.norm(loaded_embeddings, axis=1), 1.0)}"
    )

    # Usage instructions
    logger.info("\n📋 Usage Instructions:")
    logger.info(
        "1. Copy the generated NPY file to replace the current BioCLIP embeddings:"
    )
    logger.info(
        f"   cp {embeddings_filename} /path/to/bioclip/embeddings/text_vectors.npy"
    )
    logger.info("2. Restart BioCLIP service to use the new 768-dimensional embeddings")
    logger.info("3. The dimension mismatch error should now be resolved!")


if __name__ == "__main__":
    main()
