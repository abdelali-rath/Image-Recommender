import numpy as np
import torch
from PIL import Image

from image_recommender.similarity.similarity_embedding import (
    compute_clip_embedding,
    compute_clip_embeddings_batch,
    EMBEDDING_DIM,
    get_clip_model,
)

def make_imgs():
    return [
        Image.new("RGB", (96, 96), (255, 0, 0)),
        Image.new("RGB", (96, 96), (0, 255, 0)),
        Image.new("RGB", (96, 96), (0, 0, 255)),
    ]

def test_batch_shape_and_dtype():
    embs = compute_clip_embeddings_batch(make_imgs())
    assert embs.shape == (3, EMBEDDING_DIM)
    assert embs.dtype == torch.float32

def test_empty_batch():
    embs = compute_clip_embeddings_batch([])
    assert embs.shape == (0, EMBEDDING_DIM)

def test_batch_matches_single():
    imgs = make_imgs()
    get_clip_model()  # ensure single load
    singles = [compute_clip_embedding(img).numpy() for img in imgs]
    batch = compute_clip_embeddings_batch(imgs).numpy()
    for i in range(len(imgs)):
        assert np.allclose(singles[i], batch[i], atol=1e-6, rtol=1e-6)

def test_norm_is_one():
    embs = compute_clip_embeddings_batch(make_imgs()).numpy()
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6, rtol=1e-6)
